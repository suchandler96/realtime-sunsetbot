"""IMAP polling, SMTP sending and message helpers."""
import imaplib
import os
import re
import smtplib
import time
from email import message_from_bytes, policy
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr


def password(section):
    return os.environ.get(section["password_env"], "")


imaplib.Commands["ID"] = ("AUTH",)   # RFC 2971, not in imaplib's table


def _connect(imap):
    """Logged-in IMAP connection with the selected folder.

    163.com / 126.com refuse SELECT ("Unsafe Login") from clients that have not identified
    themselves with the ID command, so it is always sent after login.
    """
    conn = imaplib.IMAP4_SSL(imap["host"], imap.get("port", 993))
    conn.login(imap["user"], password(imap))
    try:
        conn._simple_command("ID", '("name" "sunsetbot" "version" "1.0")')
    except imaplib.IMAP4.error:
        pass  # servers without ID support
    conn.select(imap.get("folder", "INBOX"))
    return conn


def fetch_unseen(cfg):
    """[(imap id, EmailMessage)] for the unseen mails of the folder (not yet marked as seen)."""
    conn = _connect(cfg["imap"])
    try:
        _, ids = conn.search(None, "UNSEEN")
        out = []
        for num in ids[0].split():
            _, data = conn.fetch(num, "(BODY.PEEK[])")
            out.append((num, message_from_bytes(data[0][1], policy=policy.default)))
        return out
    finally:
        try:
            conn.logout()
        except imaplib.IMAP4.error:
            pass


def mark_seen(cfg, nums):
    if not nums:
        return
    conn = _connect(cfg["imap"])
    try:
        for num in nums:
            conn.store(num, "+FLAGS", "\\Seen")
    finally:
        conn.logout()


def sender_address(msg):
    return parseaddr(str(msg.get("Reply-To") or msg.get("From") or ""))[1].lower()


def body_text(msg):
    part = msg.get_body(("plain", "html"))
    if part is None:
        return ""
    text = part.get_content()
    if part.get_content_type() == "text/html":
        text = re.sub(r"<(script|style).*?</\1>|<[^>]+>", " ", text, flags=re.S | re.I)
    return text


def is_automatic(msg, own_address):
    """Bounces, mailing lists and our own mail: never answered (avoids reply loops)."""
    sender = sender_address(msg)
    if str(msg.get("Auto-Submitted", "no")).lower() != "no":
        return True
    if str(msg.get("Precedence", "")).lower() in ("bulk", "list", "junk"):
        return True
    return sender == own_address.lower() or re.match(r"(no-?reply|mailer-daemon|postmaster)", sender) is not None


def authorized(msg, allowed, require_dkim):
    sender = sender_address(msg)
    if not any(sender == a.lower() or (a.startswith("@") and sender.endswith(a.lower())) for a in allowed):
        return False
    return not require_dkim or "dkim=pass" in str(msg.get("Authentication-Results", "")).lower()


def build_reply(original, from_address, text, figures=()):
    reply = EmailMessage()
    subject = str(original.get("Subject", "")).strip()
    reply["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
    reply["From"], reply["To"] = from_address, sender_address(original)
    reply["Date"], reply["Message-ID"] = formatdate(localtime=True), make_msgid()
    reply["Auto-Submitted"] = "auto-replied"
    if original.get("Message-ID"):
        reply["In-Reply-To"] = reply["References"] = str(original["Message-ID"])
    reply.set_content(text)
    for path in figures:
        with open(path, "rb") as f:
            reply.add_attachment(f.read(), maintype="image", subtype="png", filename=os.path.basename(path))
    return reply


def _send_once(smtp, port, message):
    implicit_tls = port in (465, 994)  # 994 is 163.com's second implicit-TLS port
    # a fixed EHLO name: some servers drop the connection on odd (e.g. non-ASCII) Windows host names
    client = (smtplib.SMTP_SSL if implicit_tls else smtplib.SMTP)(smtp["host"], port, timeout=60,
                                                                  local_hostname="localhost")
    with client:
        client.set_debuglevel(1 if smtp.get("debug") else 0)
        if not implicit_tls:
            client.starttls()
        client.login(smtp["user"], password(smtp))
        client.send_message(message)


def send(cfg, message, attempts=3):
    """Send via the configured port; connection drops are retried, then the other implicit-TLS port is tried."""
    smtp = cfg["smtp"]
    port = smtp.get("port", 465)
    ports = [port] + [p for p in (465, 994) if p != port and port in (465, 994)]
    error = None
    for candidate in ports:
        for attempt in range(1, attempts + 1):
            try:
                return _send_once(smtp, candidate, message)
            except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, OSError) as exc:
                error = exc
                print(f"SMTP port {candidate}, attempt {attempt}/{attempts}: {type(exc).__name__}: {exc}")
                time.sleep(5 * attempt)
            # authentication / recipient / content refusals are not transient: they raise to the caller
    raise error
