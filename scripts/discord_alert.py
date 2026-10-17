#!/usr/bin/env python3
"""Send a Discord alert using bot token and channel id from .env."""

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
    "DISCORD_BOT_TOKEN",
    "BOT_TOKEN",
)
CHANNEL_ID_KEYS = (
    "DISCORD_CHANNEL_ID",
    "CHANNEL_ID",
)

DISCORD_API_BASE = "https://discord.com/api/v10"


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


DISCORD_MAX_CONTENT = 2000


def send_discord_message(token: str, channel_id: str, text: str) -> None:
    url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
    if len(text) > DISCORD_MAX_CONTENT:
        text = text[: DISCORD_MAX_CONTENT - 3] + "..."
    payload = json.dumps({"content": text}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bot {token}",
            "User-Agent": "DiscordBot (https://github.com/discord/discord-api-docs, 10)",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        if "id" not in parsed:
            raise RuntimeError(f"Discord API error: {body}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send Discord alert using bot token/channel id from .env",
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
        "--skip-session-alert",
        action="store_true",
        help="Skip alerts when job name includes 'session'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(resolve_env_file(args.env_file))

    token = first_env(TOKEN_KEYS)
    channel_id = first_env(CHANNEL_ID_KEYS)

    if not token:
        print(
            "Missing Discord bot token. Set one of: " + ", ".join(TOKEN_KEYS),
            file=sys.stderr,
        )
        return 2

    if not channel_id:
        print(
            "Missing Discord channel id. Set one of: " + ", ".join(CHANNEL_ID_KEYS),
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
        send_discord_message(token, channel_id, message)
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Failed to send Discord alert: {exc}", file=sys.stderr)
        return 1

    print("Discord alert sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
