#!/usr/bin/env python3
"""Send a Telegram alert using bot token and chat id from .env."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

TOKEN_KEYS = (
    "SERVER_ALERT_TELEGRAM_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "BOT_TOKEN",
)
CHAT_ID_KEYS = (
    "TELEGRAM_ID",
    "TELEGRAM_CHAT_ID",
    "CHAT_ID",
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_env_file(path_arg: str) -> Path:
    requested = Path(path_arg)
    if requested.is_absolute():
        return requested

    cwd_candidate = Path.cwd() / requested
    if cwd_candidate.exists():
        return cwd_candidate

    script_candidate = Path(__file__).resolve().parent.parent / requested
    return script_candidate


def first_env(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def resolve_folder_name(cwd: str) -> str:
    folder = Path(cwd).resolve().name
    if folder:
        return folder
    return cwd


def build_alert_message(base_text: str, status: str, summary: str) -> str:
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    host = socket.gethostname()
    cwd = os.getcwd()
    folder = resolve_folder_name(cwd)
    lines = [
        base_text.strip(),
        "",
        f"Server: {host}",
        f"Folder: {folder}",
        f"Summary: {summary}",
        f"Status: {status}",
        f"Time: {timestamp}",
        f"Path: {cwd}",
    ]
    return "\n".join(lines)


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        if not parsed.get("ok", False):
            raise RuntimeError(f"Telegram API error: {body}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send Telegram alert using token/chat id from .env",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--message",
        default="",
        help="Custom alert message. If omitted, a default completion message is used.",
    )
    parser.add_argument(
        "--status",
        default="DONE",
        help="Status label used in default message (default: DONE)",
    )
    parser.add_argument(
        "--job",
        default="job",
        help="Job/session name used in default message (default: job)",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Short summary of completed work (default: same as --job).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the message and exit without sending.",
    )
    parser.add_argument(
        "--allow-session-alert",
        action="store_true",
        help="Deprecated compatibility flag; session alerts are allowed by default.",
    )
    parser.add_argument(
        "--skip-session-alert",
        action="store_true",
        help="Skip alerts when job name includes 'session'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(resolve_env_file(args.env_file))

    token = first_env(TOKEN_KEYS)
    chat_id = first_env(CHAT_ID_KEYS)

    if not token:
        print(
            "Missing Telegram bot token. Set one of: "
            + ", ".join(TOKEN_KEYS),
            file=sys.stderr,
        )
        return 2

    if not chat_id:
        print(
            "Missing Telegram chat id. Set one of: "
            + ", ".join(CHAT_ID_KEYS),
            file=sys.stderr,
        )
        return 2

    summary = args.summary if args.summary else args.job
    if args.message:
        message = build_alert_message(args.message, args.status, summary)
    else:
        default_text = f"[{args.status}] {args.job} finished."
        message = build_alert_message(default_text, args.status, summary)

    if ("session" in args.job.lower()) and args.skip_session_alert:
        print("Session alert skipped by policy.")
        return 0

    if args.dry_run:
        print(message)
        return 0

    try:
        send_telegram_message(token, chat_id, message)
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Failed to send Telegram alert: {exc}", file=sys.stderr)
        return 1

    print("Telegram alert sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
