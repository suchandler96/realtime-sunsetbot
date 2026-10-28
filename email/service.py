"""E-mail enquiry service for main.py.

    python email/service.py                    # poll the mailbox forever
    python email/service.py --once             # answer what is waiting, then exit
    python email/service.py --text "city: Tokyo"   # show how a request is parsed
    python email/service.py --eml request.eml  # answer a saved mail; reply is written to <name>.reply.eml

Do not add an __init__.py to this folder: the stdlib `email` package must keep winning the import.
"""
import argparse
import os
import sys
import time
import tomllib
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mailio  # noqa: E402
import request_parser as rp  # noqa: E402
import runner  # noqa: E402


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def llm_settings(cfg):
    llm = cfg.get("llm", {})
    key = os.environ.get(llm.get("api_key_env", ""), "")
    return {"api_key": key, "model": llm.get("model", "claude-haiku-5-5")} if key else None


def answer(cfg, msg):
    """Reply text and figure paths for one request message."""
    service = cfg.get("service", {})
    policy = rp.Policy(allow_satellite=service.get("allow_satellite", False))
    request = rp.parse_request(str(msg.get("Subject", "")), mailio.body_text(msg), policy, llm_settings(cfg))
    if request.problems:
        return "I could not run this request:\n  - " + "\n  - ".join(request.problems) + "\n\n" + rp.USAGE, []
    silent = service.get("silent", False)
    argv = rp.to_argv(request.options, silent)
    print("running main.py", " ".join(argv))
    result = runner.run_main(argv, service.get("run_timeout_seconds", 900), service.get("max_output_chars", 20000),
                             headless=silent)
    lines = ["Interpreted as: " + " ".join(a for a in argv if a not in ("--save", "--silent")), *request.notes, ""]
    lines.append(f"Finished in {result.seconds:.0f} s." if result.ok else "The run failed; output below.")
    return "\n".join(lines) + "\n\n" + result.output, result.figures


class RateLimit:
    def __init__(self, per_hour):
        self.per_hour, self.seen = per_hour, defaultdict(deque)

    def allow(self, sender):
        now, queue = time.time(), self.seen[sender]
        while queue and now - queue[0] > 3600:
            queue.popleft()
        if len(queue) >= self.per_hour:
            return False
        queue.append(now)
        return True


_replies = {}  # Message-ID -> reply built but not yet delivered, so a failed send never re-runs main.py


def handle(cfg, msg, limiter, dry_run_path=None):
    """Process one message; returns True when it was consumed (answered or deliberately ignored)."""
    service = cfg.get("service", {})
    own = cfg["smtp"]["user"]
    sender = mailio.sender_address(msg)
    if mailio.is_automatic(msg, own) or not mailio.authorized(msg, service.get("allowed_senders", []),
                                                              service.get("require_dkim_pass", False)):
        print(f"ignored mail from {sender or '?'}")
        return True
    key = str(msg.get("Message-ID") or "")
    reply = _replies.get(key) if key else None
    if reply is None:
        if not limiter.allow(sender):
            text, figures = "Too many requests this hour; please try again later.", []
        else:
            try:
                text, figures = answer(cfg, msg)
            except Exception as exc:  # keep the service alive whatever one request does
                print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
                text, figures = "Sorry, something went wrong while handling your request.", []
        reply = mailio.build_reply(msg, own, text, figures)
        if key:
            _replies[key] = reply
    if dry_run_path:
        with open(dry_run_path, "wb") as f:
            f.write(bytes(reply))
        print(f"reply written to {dry_run_path}")
    else:
        mailio.send(cfg, reply)
        _replies.pop(key, None)
        print(f"answered {sender}")
    return True


def poll_once(cfg, limiter):
    done = []
    for num, msg in mailio.fetch_unseen(cfg):
        try:
            if handle(cfg, msg, limiter):
                done.append(num)
        except Exception as exc:  # e.g. SMTP down: leave the mail unseen to retry next round
            print(f"could not answer: {type(exc).__name__}: {exc}", file=sys.stderr)
    mailio.mark_seen(cfg, done)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.toml"))
    ap.add_argument("--once", action="store_true", help="process the waiting mails and exit")
    ap.add_argument("--text", help="only show how this request text is parsed (nothing is run)")
    ap.add_argument("--eml", help="answer this saved message instead of polling; the reply is written next to it")
    args = ap.parse_args()

    if args.text is not None:
        cfg = load_config(args.config) if os.path.exists(args.config) else {}
        policy = rp.Policy(cfg.get("service", {}).get("allow_satellite", False))
        req = rp.parse_request("", args.text, policy, llm_settings(cfg))
        print("problems:", req.problems or "none")
        print("argv:", " ".join(rp.to_argv(req.options, cfg.get("service", {}).get("silent", False))))
        return
    cfg = load_config(args.config)
    limiter = RateLimit(cfg.get("service", {}).get("max_requests_per_hour", 6))
    if args.eml:
        from email import message_from_binary_file, policy
        with open(args.eml, "rb") as f:
            msg = message_from_binary_file(f, policy=policy.default)
        handle(cfg, msg, limiter, dry_run_path=os.path.splitext(args.eml)[0] + ".reply.eml")
        return
    if not cfg.get("service", {}).get("allowed_senders"):
        sys.exit("config: service.allowed_senders must list at least one sender")
    while True:
        try:
            poll_once(cfg, limiter)
        except Exception as exc:  # network / IMAP hiccup
            print(f"poll failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.once:
            break
        time.sleep(cfg.get("service", {}).get("poll_seconds", 60))


if __name__ == "__main__":
    main()
