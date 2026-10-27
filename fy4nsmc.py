"""Download the newest FY-4B AGRI L2 files from the NSMC data portal.

Replaces the basket/order flow of ``fy4download.py`` (data.cma.cn) with the
simpler one on https://data.nsmc.org.cn/DataPortal: the web page's own steps are

    filter FY-4B -> product (e.g. cloud top height) -> search (default time filter)
    -> click the download icon of the newest file in the pop-up list

and this script does the same through the portal's JSON API.  Searching needs no
login; downloading does, with a verification code on login that **you type** (the
captcha image is saved to a file; nothing is solved automatically).  The login
session is cached so later runs usually need no code at all.

    python fy4nsmc.py                 # newest CTH and CTT file
    python fy4nsmc.py --product CLP   # any AGRI L2 product code (CLP, CLT, CLM, CTP, ...)
    python fy4nsmc.py --count 3       # newest three files of each product

If the download of the newest file stalls or fails (the browser's download page
shows nothing), the next newest file is tried, as you would do by hand.
"""

import argparse
import base64
import getpass
import json
import os
import re
import secrets
import sys
import time
import uuid
from html import unescape as html_unescape
from datetime import datetime, timedelta, timezone

import requests

BASE = "https://data.nsmc.org.cn/DataPortal"
USER_AGENT = "fy4nsmc/1.0 (personal data download script)"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SESSION_FILE = os.path.join(HERE, ".nsmc_session.json")
# Picked up automatically when it exists: username on line 1, password on line 2.
DEFAULT_CREDENTIALS_FILE = os.path.join(HERE, "nsmc_credentials.txt")
DEFAULT_PRODUCTS = ["CTH", "CTT"]  # cloud top height, cloud top temperature
SATELLITE, INSTRUMENT, AREA = "FY4B", "AGRI", "DISK"
DOWNLOAD_SOURCE = "NewPortalCH"  # what the Chinese web page sends


# ---- RSA (PKCS#1 v1.5) for the sign-in page's password field -------------------
def _der(data, pos=0):
    """Read one DER element at ``pos``; returns (tag, value_bytes, next_pos)."""
    tag, length = data[pos], data[pos + 1]
    pos += 2
    if length & 0x80:
        n = length & 0x7F
        length = int.from_bytes(data[pos:pos + n], "big")
        pos += n
    return tag, data[pos:pos + length], pos + length


def rsa_public_key(b64url):
    """(n, e) from a base64url SubjectPublicKeyInfo as shipped on the sign-in page."""
    der = base64.urlsafe_b64decode(b64url + "=" * (-len(b64url) % 4))
    _, spki, _ = _der(der)
    _, _, pos = _der(spki)  # algorithm identifier
    _, bitstring, _ = _der(spki, pos)
    _, rsa_key, _ = _der(bitstring[1:])  # skip the unused-bits byte
    _, n, pos = _der(rsa_key)
    _, e, _ = _der(rsa_key, pos)
    return int.from_bytes(n, "big"), int.from_bytes(e, "big")


def rsa_encrypt_b64(b64url_key, text):
    """What the page's JSEncrypt does: RSA with PKCS#1 v1.5 padding, base64 output."""
    n, e = rsa_public_key(b64url_key)
    k = (n.bit_length() + 7) // 8
    message = text.encode()
    if len(message) > k - 11:
        raise PortalError("password too long for the portal's key")
    padding = bytes(secrets.randbelow(255) + 1 for _ in range(k - 3 - len(message)))  # non-zero bytes
    block = b"\x00\x02" + padding + b"\x00" + message
    cipher = pow(int.from_bytes(block, "big"), e, n)
    return base64.b64encode(cipher.to_bytes(k, "big")).decode()


def _input_value(html, element_id):
    m = re.search(r'<input[^>]*id="%s"[^>]*value="([^"]*)"' % re.escape(element_id), html)
    return m.group(1) if m else ""


class PortalError(RuntimeError):
    pass


class NotLoggedIn(PortalError):
    pass


class TooFrequent(PortalError):
    pass


class Portal:
    def __init__(self, captcha_path="nsmc_captcha.jpg", answer_timeout=300):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.captcha_path = captcha_path
        self.captcha_base = os.path.splitext(captcha_path)[0]
        self.answer_timeout = answer_timeout

    def _ask_code(self):
        """The verification code, typed by a human: at the prompt."""
        return input(f"Verification code (open {os.path.abspath(self.captcha_path)}; "
                     "the page expires quickly, so answer promptly): ").strip()

    # -- generic -----------------------------------------------------------
    def get_json(self, path, **kw):
        r = self.s.get(BASE + path, timeout=60, **kw)
        r.raise_for_status()
        return r.json()

    def csrf_headers(self):
        """Search calls carry the Csrf-Token the page fetches first."""
        token = self.get_json("/v1/data/selection/token").get("resource")
        return {"Csrf-Token": token} if token else {}

    # -- products ----------------------------------------------------------
    def product_code(self, datatype):
        """File-name code of the full-disk FY-4B AGRI product for a data type such as CTH."""
        path = (f"/v1/data/type/satellite/{SATELLITE}/instrument/{INSTRUMENT}"
                f"/datatype/{datatype}/areatype/{AREA}/product")
        products = self.get_json(path).get("resource") or []
        if not products:
            raise PortalError(f"no {SATELLITE} {INSTRUMENT} {datatype} product on the portal")
        if len(products) > 1:
            print(f"{datatype}: {len(products)} products, using the first:")
            for p in products:
                print("   ", p.get("productionnameeng"), p["filenamecode"])
        return products[0]["filenamecode"], products[0].get("productionnameeng") or datatype

    # -- search ------------------------------------------------------------
    def search(self, product_code, begin, end, limit):
        """Files of a product between two UTC datetimes, newest first."""
        params = {
            "productID": product_code,
            "txtBeginDate": begin.strftime("%Y-%m-%d"), "txtBeginTime": begin.strftime("%H:%M:%S"),
            "txtEndDate": end.strftime("%Y-%m-%d"), "txtEndTime": end.strftime("%H:%M:%S"),
            "beginIndex": 1, "endIndex": limit, "where": "", "timeSelection": "all",
            "periodTime": "", "daynight": "", "source": 0,
            "cbAllArea": "true", "converStatus": "", "rdbIsEvery": "",
        }
        result = self.get_json("/v1/data/selection/subfile", params=params, headers=self.csrf_headers())
        rows = result.get("resource") or []
        rows.sort(key=lambda r: str(r.get("DATABEGINDATE")), reverse=True)  # page shows newest on top
        return rows

    # -- login -------------------------------------------------------------
    def logged_in(self):
        try:
            return self.get_json("/v1/data/user/status/login").get("status") == 1
        except (requests.RequestException, ValueError):
            return False

    def _https(self, url):
        """The portal redirects to plain http; keep credentials and cookies on https."""
        return "https://" + url[len("http://"):] if url.startswith("http://") else url

    def _signin_page(self):
        """Open the portal's login URL and return (form fields, auth-centre base URL).

        The page's login key and public key expire after a short time, so this is repeated for
        every attempt, immediately before the captcha is shown.
        """
        page = self.s.get(BASE + "/v1/data/user/login", timeout=60, allow_redirects=False,
                          params={"newurl": BASE + "/cn/data/detail.html"})
        while page.is_redirect:  # follow by hand so every hop stays on https
            page = self.s.get(self._https(page.headers["Location"]), timeout=60, allow_redirects=False)
        page.raise_for_status()
        form = {name: html_unescape(_input_value(page.text, field))
                for name, field in (("lk", "inputLoginKeyCN"), ("rd", "inputSourceURLCN"), ("key", "keyCN"))}
        if not all(form.values()):
            raise PortalError("could not find the login key / public key on the sign-in page")
        return form, page.url.split("/v1/user/login")[0] + "/v1/user"  # https://.../authcenter/v1/user

    def login(self, username, password, attempts=4):
        """Sign in through the portal's auth centre (the same page the browser is sent to)."""
        for attempt in range(1, attempts + 1):
            form, auth = self._signin_page()  # fresh page (and keys) for every attempt
            captcha_id = str(uuid.uuid4())
            img = self.s.get(auth + "/validateCode", timeout=60,
                             params={"captchaId": captcha_id, "data": int(time.time() * 1000) % 1048576})
            img.raise_for_status()
            # A new file name per attempt, so an image viewer can never show a stale picture.
            self.captcha_path = f"{self.captcha_base}_{attempt}.jpg"
            with open(self.captcha_path, "wb") as f:
                f.write(img.content)
            shown = time.time()
            code = self._ask_code()
            print(f"answered {time.time() - shown:.0f}s after the page was loaded")
            res = self.s.post(auth + "/commit", timeout=60, json={
                "userName": username, "thePassword": rsa_encrypt_b64(form["key"], password),
                "validateCode": code, "loginKey": form["lk"], "sourceURL": form["rd"], "captchaId": captcha_id})
            res.raise_for_status()
            body = res.json()
            if body.get("status") == 1:
                target = (body.get("resource") or {}).get("sourceURL")
                if target:  # completes the single sign-on: sets the DataPortal session cookie
                    self.s.get(self._https(target), timeout=60)
                if not self.logged_in():
                    raise PortalError("login succeeded at the auth centre but the portal session is not logged in")
                print("Logged in as", username)
                return
            print(f"Login refused ({time.time() - shown:.0f}s after the page was loaded):", body.get("message"))
        raise PortalError("login failed")

    def load_session(self, path):
        """Reuse cookies saved by ``save_session`` while the portal still considers them logged in."""
        try:
            with open(path) as f:
                saved = json.load(f)["cookies"]
        except (OSError, ValueError, KeyError):
            return False
        self.s.cookies.clear()  # drop the anonymous session cookies set by earlier calls
        if isinstance(saved, dict):  # older file format: name -> value only
            saved = [{"name": k, "value": v} for k, v in saved.items()]
        for c in saved:
            self.s.cookies.set(c["name"], c["value"], domain=c.get("domain", ""), path=c.get("path", "/"))
        return self.logged_in()

    def save_session(self, path):
        """Keep each cookie with its domain and path (the auth centre and the portal share names)."""
        cookies = [{"name": c.name, "value": c.value, "domain": c.domain, "path": c.path} for c in self.s.cookies]
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"cookies": cookies}, f)
        os.chmod(path, 0o600)

    # -- download ----------------------------------------------------------
    def open_download(self, row):
        """Start the download of a file as the page's download icon does; returns a streaming response."""
        name = row["ARCHIVENAME"]
        status = self.get_json(f"/v1/data/selection/file/{name}/download/status",
                               params={"downloadSource": DOWNLOAD_SOURCE})
        code = status.get("status")
        if code == -2:
            raise NotLoggedIn("the portal says you are not logged in")
        if code in (-1, -3):
            raise TooFrequent("the portal says you download too often; try again later")
        if code is None or code < 0:
            raise PortalError(f"download status {status}")
        url = status.get("resource")
        if url and "http" in url and name in url:
            return self.s.get(url, stream=True, timeout=(15, 60))
        return self.s.post(BASE + "/v1/data/selection/file/download", stream=True, timeout=(15, 60), data={
            "downloadSource": DOWNLOAD_SOURCE, "fileName": name, "centerFlag": row.get("CNETERFLAG", "")})

    def download(self, row, out_dir, deadline_s):
        """Download one file; returns its path, or raises on any failure (partial file removed)."""
        name = row["ARCHIVENAME"]
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, name)
        part = path + ".part"
        resp = self.open_download(row)
        try:
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype or "html" in ctype:
                raise PortalError(f"server answered with {ctype}: {resp.text[:200]!r}")
            started = time.time()
            with open(part, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
                    if time.time() - started > deadline_s:
                        raise PortalError(f"download exceeded {deadline_s}s")
            size, expected = os.path.getsize(part), int(float(row.get("DATASIZE") or 0))
            if expected and size != expected:
                raise PortalError(f"size {size} != expected {expected}")
            os.replace(part, path)
            return path
        except BaseException:
            if os.path.exists(part):
                os.remove(part)
            raise
        finally:
            resp.close()


def read_credentials(path):
    """User name and password from a two-line text file (kept out of git; delete it when done)."""
    with open(path) as f:
        lines = [ln.rstrip("\r\n") for ln in f]
    if len(lines) < 2 or not lines[0] or not lines[1]:
        raise PortalError(f"{path} must contain the user name on line 1 and the password on line 2")
    return lines[0], lines[1]


def fetch_newest(products, out_dir="fy4b_nsmc", count=1, days=1, candidates=10, file_timeout=30,
                 session_file=DEFAULT_SESSION_FILE, use_session_cache=True, search_only=False, portal=None,
                 credentials_file=None):
    """Download the newest ``count`` files of each product; returns ``({product: [paths]}, n_failed)``.

    Files already present (same size) are reused without logging in.  Logging in happens only when
    something has to be downloaded: cached cookies first, otherwise an interactive prompt.
    """
    now = datetime.now(timezone.utc)
    begin = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=23, minute=59, second=59, microsecond=0)

    portal = portal or Portal()
    found, failures = {}, 0
    for datatype in products:
        found[datatype] = []
        code, title = portal.product_code(datatype)
        rows = portal.search(code, begin, end, max(candidates, count))
        print(f"\n{title}: {len(rows)} newest file(s) {begin:%Y-%m-%d} .. {end:%Y-%m-%d} UTC")
        for row in rows[:5]:
            print(f"   {row['DATABEGINDATE']}  {int(float(row['DATASIZE'])) / 1e6:6.1f} MB  {row['ARCHIVENAME']}")
        if not rows or search_only:
            continue

        product_dir = os.path.join(out_dir, datatype)
        for row in rows:  # newest first; fall through to older files when one fails
            if len(found[datatype]) >= count:
                break
            local = os.path.join(product_dir, row["ARCHIVENAME"])
            if os.path.exists(local) and os.path.getsize(local) == int(float(row.get("DATASIZE") or -1)):
                print("already downloaded", local)
                found[datatype].append(local)
                continue
            if not portal.logged_in():
                cached = use_session_cache and portal.load_session(session_file)
                if not cached:
                    credentials_file = credentials_file or (
                        DEFAULT_CREDENTIALS_FILE if os.path.isfile(DEFAULT_CREDENTIALS_FILE) else None)
                    if credentials_file:
                        print(f"Using login credentials from {credentials_file}")
                        username, password = read_credentials(credentials_file)
                    else:
                        username = os.environ.get("NSMC_USER") or input("data.nsmc.org.cn user name: ").strip()
                        password = os.environ.get("NSMC_PASS") or getpass.getpass("password: ")
                    portal.login(username, password)
                    if use_session_cache:
                        portal.save_session(session_file)
            try:
                path = portal.download(row, product_dir, file_timeout)
            except TooFrequent as exc:
                print("stop:", exc)
                break
            except (PortalError, requests.RequestException) as exc:
                print(f"failed {row['ARCHIVENAME']}: {exc}; trying the next newest file")
                continue
            print("saved", path)
            found[datatype].append(path)
        if len(found[datatype]) < count and not search_only:
            failures += 1
            print(f"only {len(found[datatype])} of {count} file(s) available for {datatype}")
    return found, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--product", nargs="+", default=DEFAULT_PRODUCTS,
                    help="FY-4B AGRI L2 data types, e.g. CTH CTT CLP CLT CLM (default: CTH CTT)")
    ap.add_argument("--count", type=int, default=1, help="how many of the newest files to download per product")
    ap.add_argument("--candidates", type=int, default=10,
                    help="how many newest files to list; later ones are fallbacks when a download fails")
    ap.add_argument("--days", type=int, default=1,
                    help="search window: from 00:00:00 UTC this many days ago to 23:59:59 UTC today "
                         "(the page's default is yesterday..today)")
    ap.add_argument("--out", default="fy4b_nsmc", help="download directory")
    ap.add_argument("--file-timeout", type=int, default=30, help="give up on a file after this many seconds")
    ap.add_argument("--search-only", action="store_true", help="list the files, do not download")
    ap.add_argument("--session-file", default=DEFAULT_SESSION_FILE, help="cached login cookies")
    ap.add_argument("--no-session-cache", action="store_true", help="do not read or write the session file")
    ap.add_argument("--credentials-file", help="text file: user name on line 1, password on line 2 "
                                               "(default: nsmc_credentials.txt next to this script, if it exists; else NSMC_USER/NSMC_PASS or a prompt)")
    args = ap.parse_args()

    _, failures = fetch_newest(
        args.product, args.out, args.count, args.days, args.candidates, args.file_timeout,
        args.session_file, not args.no_session_cache, args.search_only,
        credentials_file=args.credentials_file)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
