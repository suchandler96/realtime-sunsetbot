# E-mail enquiry service

Reads requests from a mailbox, runs `main.py` and mails back its text output and the figure(s).

    cp email/config.example.toml email/config.toml     # edit; the file is git-ignored
    export SUNSETBOT_MAIL_PASSWORD=...                 # mailbox password / app password
    .venv_sunsetbot/bin/python email/service.py        # poll forever (--once: single pass)

Requests are `key: value` lines (see `USAGE` in `request_parser.py`):

    city: Shanghai
    date: 2026-11-02
    event: sunset
    radar: yes

Free-text understanding (English/Chinese keywords, and Claude via `ANTHROPIC_API_KEY`) is implemented but switched
off by `NATURAL_LANGUAGE = False` in `request_parser.py`.

Try it without a mailbox:

    .venv_sunsetbot/bin/python email/service.py --text "city: Tokyo"                 # parse only
    .venv_sunsetbot/bin/python email/service.py --eml request.eml                    # full run, reply -> request.reply.eml

Files: `service.py` (loop, authorisation, rate limit), `request_parser.py` (text -> validated options -> argv),
`runner.py` (runs main.py, collects figures into `email/jobs/`), `mailio.py` (IMAP/SMTP).

Safety model
- Only `allowed_senders` get an answer (others are ignored silently); `require_dkim_pass` guards against forged From.
- Mail text is never passed to a shell. Whatever the parser (rules or LLM) returns is whitelisted and range-checked
  (known cities or lat/lon, date within -3..+7 days, fixed model list, ...); `main.py` is started with an argv list,
  a timeout and no stdin. File-path options (`--satellite-file`, ...) cannot be set from mail.
- `--satellite` (FY-4B download with NSMC login) is off unless `allow_satellite = true`.
- Requests run one at a time; per-sender hourly limit; automatic mail (bounces, lists, own address) is never answered.
- Do not add `email/__init__.py`: the folder name equals a stdlib package and must stay a plain directory.