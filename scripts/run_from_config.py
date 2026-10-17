#!/usr/bin/env python3
"""Translate a resolved config dump (scripts/configs/*_tmal_test.yaml) into a
`src/engine/train.py` CLI invocation and run it.

These yaml files are `train.py`'s own `config.yaml` dump (see `--save` handling
in `src/engine/train.py`), not launcher input, so this script maps each
top-level key back to its argparse flag and re-invokes `train.py` directly.
Nested/derived keys (`save`, `connectivity_layout`, `conn_channels`,
`monitoring_config`, `early_stopping_config`, `checkpoint_config`) are skipped
since `train.py` recomputes them from the flat flags.

Usage:
    scripts/run_from_config.py --config scripts/configs/drive_tmal_test.yaml --mode train
    scripts/run_from_config.py --config scripts/configs/drive_tmal_test.yaml --mode test \\
        --pretrained ckpts/drive_model.pth
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"

# yaml key -> CLI flag name, for the handful of flags that use a dash instead
# of the key's underscore.
DASH_FLAGS = {
    "num_class": "num-class",
    "batch_size": "batch-size",
    "lr_update": "lr-update",
    "lr_step": "lr-step",
    "save_per_epochs": "save-per-epochs",
}

# store_true flags: emit only when true (their argparse default is false).
BOOL_FLAGS = {"use_seg_aux", "debug_loss_breakdown", "simple_loss"}

# keys `train.py` derives itself and writes back into its config dump; not
# valid input flags.
DERIVED_KEYS = {
    "save", "connectivity_layout", "conn_channels",
    "monitoring_config", "early_stopping_config", "checkpoint_config",
}

# handled separately (mode / checkpoint), not passed through verbatim.
MODE_KEYS = {"test_only", "pretrained"}

# already-true by argparse default in train.py; only worth emitting if the
# config disagrees, which none of our resolved dumps do.
DEFAULT_TRUE_KEYS = {"tie_break_with_loss"}
DEFAULT_FALSE_KEYS = {"save_best_only"}


def build_args(cfg: dict, mode: str, pretrained: str | None) -> list[str]:
    args: list[str] = []
    for key, value in cfg.items():
        if key in DERIVED_KEYS or key in MODE_KEYS:
            continue
        if value is None:
            continue
        if key in DEFAULT_TRUE_KEYS and value is True:
            continue
        if key in DEFAULT_FALSE_KEYS and value is False:
            continue
        if key in BOOL_FLAGS:
            if value:
                args.append(f"--{key}")
            continue
        if key == "resize":
            args += ["--resize", str(value[0]), str(value[1])]
            continue
        flag = DASH_FLAGS.get(key, key)
        args += [f"--{flag}", str(value)]

    if mode == "test":
        if not pretrained:
            raise SystemExit("--mode test requires --pretrained <checkpoint path>")
        args += ["--pretrained", pretrained, "--test_only"]
    elif pretrained:
        args += ["--pretrained", pretrained]

    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="resolved config yaml (e.g. scripts/configs/drive_tmal_test.yaml)")
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--pretrained", type=str, default=None, help="checkpoint path (required for --mode test)")
    p.add_argument("--dry_run", action="store_true", help="print the resulting train.py command instead of running it")
    p.add_argument("overrides", nargs=argparse.REMAINDER,
                    help="extra flags appended verbatim to override the config (e.g. --device 1 --max_eval_batches 3)")
    return p.parse_args()


def main() -> int:
    ns = parse_args()
    cfg = yaml.safe_load(ns.config.read_text())

    overrides = ns.overrides[1:] if ns.overrides[:1] == ["--"] else ns.overrides
    args = build_args(cfg, ns.mode, ns.pretrained)
    cmd = [str(PYTHON_BIN), str(REPO_ROOT / "src" / "engine" / "train.py"), *args, *overrides]

    print(" \\\n  ".join(cmd))
    if ns.dry_run:
        return 0

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
