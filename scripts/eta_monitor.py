#!/usr/bin/env python3
"""Estimate training ETA from a results.csv file."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class EtaSnapshot:
    csv_path: Path
    checked_at: datetime
    status: str
    last_epoch: int
    total_epochs: int
    remaining_epochs: int
    elapsed_seconds: int
    avg_sec_per_epoch: float
    eta_seconds: float
    expected_finish_time: datetime
    valid_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate ETA from results.csv using elapsed_hms and epoch."
    )
    parser.add_argument("--csv", required=True, help="Path to results.csv")
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=500,
        help="Total training epochs used for ETA (default: 500)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh ETA until interrupted",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Refresh interval in seconds for --watch (default: 10)",
    )
    parser.add_argument(
        "--tz",
        default="Asia/Seoul",
        help="Timezone for expected finish time (default: Asia/Seoul)",
    )
    return parser.parse_args()


def parse_elapsed_hms(text: str) -> int:
    value = text.strip()
    if not value:
        raise ValueError("empty elapsed_hms")
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid elapsed_hms format: {text!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
        raise ValueError(f"invalid elapsed_hms range: {text!r}")
    return (hours * 3600) + (minutes * 60) + seconds


def read_progress(csv_path: Path) -> tuple[int, int, int]:
    if not csv_path.is_file():
        raise ValueError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing")

        required_cols = {"epoch", "elapsed_hms"}
        missing_cols = sorted(required_cols - set(reader.fieldnames))
        if missing_cols:
            raise ValueError(
                f"CSV missing required column(s): {', '.join(missing_cols)}"
            )

        latest_epoch = -1
        latest_elapsed_seconds = -1
        valid_rows = 0

        for row in reader:
            epoch_raw = (row.get("epoch") or "").strip()
            elapsed_raw = (row.get("elapsed_hms") or "").strip()
            if not epoch_raw or not elapsed_raw:
                continue

            try:
                epoch = int(epoch_raw)
                elapsed_seconds = parse_elapsed_hms(elapsed_raw)
            except ValueError:
                continue

            valid_rows += 1
            if epoch > latest_epoch or (
                epoch == latest_epoch and elapsed_seconds > latest_elapsed_seconds
            ):
                latest_epoch = epoch
                latest_elapsed_seconds = elapsed_seconds

        if valid_rows == 0:
            raise ValueError(
                "No valid rows found (need parseable epoch and elapsed_hms values)"
            )
        if latest_epoch <= 0:
            raise ValueError(f"Invalid latest epoch value: {latest_epoch}")

        return latest_epoch, latest_elapsed_seconds, valid_rows


def compute_eta(csv_path: Path, total_epochs: int, tz: ZoneInfo) -> EtaSnapshot:
    last_epoch, elapsed_seconds, valid_rows = read_progress(csv_path)
    avg_sec_per_epoch = elapsed_seconds / float(last_epoch)

    remaining_epochs = max(total_epochs - last_epoch, 0)
    eta_seconds = avg_sec_per_epoch * float(remaining_epochs)
    status = "running" if remaining_epochs > 0 else "completed_or_overrun"

    checked_at = datetime.now(tz)
    expected_finish_time = checked_at + timedelta(seconds=eta_seconds)

    return EtaSnapshot(
        csv_path=csv_path,
        checked_at=checked_at,
        status=status,
        last_epoch=last_epoch,
        total_epochs=total_epochs,
        remaining_epochs=remaining_epochs,
        elapsed_seconds=elapsed_seconds,
        avg_sec_per_epoch=avg_sec_per_epoch,
        eta_seconds=eta_seconds,
        expected_finish_time=expected_finish_time,
        valid_rows=valid_rows,
    )


def format_duration(seconds: float) -> str:
    whole_seconds = int(round(max(seconds, 0.0)))
    days, rem = divmod(whole_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_snapshot(snapshot: EtaSnapshot) -> None:
    checked_at = snapshot.checked_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    expected_finish = snapshot.expected_finish_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"[ETA] checked_at={checked_at} status={snapshot.status}")
    print(f"csv_path={snapshot.csv_path}")
    print(
        f"progress={snapshot.last_epoch}/{snapshot.total_epochs} "
        f"remaining_epochs={snapshot.remaining_epochs}"
    )
    print(f"valid_rows={snapshot.valid_rows}")
    print(f"avg_sec_per_epoch={snapshot.avg_sec_per_epoch:.2f}")
    print(f"eta_duration={format_duration(snapshot.eta_seconds)}")
    print(f"expected_finish_time={expected_finish}")
    print("")


def run_once(csv_path: Path, total_epochs: int, tz: ZoneInfo) -> None:
    snapshot = compute_eta(csv_path=csv_path, total_epochs=total_epochs, tz=tz)
    print_snapshot(snapshot)


def run_watch(csv_path: Path, total_epochs: int, interval: float, tz: ZoneInfo) -> None:
    while True:
        snapshot = compute_eta(csv_path=csv_path, total_epochs=total_epochs, tz=tz)
        print_snapshot(snapshot)
        time.sleep(interval)


def main() -> int:
    args = parse_args()

    if args.total_epochs <= 0:
        print("[ERROR] --total-epochs must be a positive integer", file=sys.stderr)
        return 2
    if args.interval <= 0:
        print("[ERROR] --interval must be greater than 0", file=sys.stderr)
        return 2

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path

    try:
        tz = ZoneInfo(args.tz)
    except ZoneInfoNotFoundError:
        print(f"[ERROR] Unknown timezone: {args.tz}", file=sys.stderr)
        return 2

    try:
        if args.watch:
            run_watch(
                csv_path=csv_path,
                total_epochs=args.total_epochs,
                interval=args.interval,
                tz=tz,
            )
        else:
            run_once(csv_path=csv_path, total_epochs=args.total_epochs, tz=tz)
    except KeyboardInterrupt:
        print("[INFO] Stopped by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
