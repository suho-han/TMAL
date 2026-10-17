"""Rename experiment output folders to canonical names.

For each folder under the given input roots, reads config.yaml, computes the
canonical experiment name via build_experiment_name_from_runtime_config(), and
renames the folder if it doesn't already match.  Also patches the
experiment_name_suffix and save fields in config.yaml.

Usage:
    uv run scripts/rename_canonical_experiment_dirs.py [--roots output/drive output/chase ...]
    uv run scripts/rename_canonical_experiment_dirs.py --dry-run     # preview only
"""

import argparse
import os
import re
import sys
from pathlib import Path

import autorootcwd  # noqa: F401 — sets cwd to repo root

_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from aggregate_results import (
    build_experiment_name_from_runtime_config,
    load_runtime_config,
)


def _find_experiment_dirs(roots: list[str]) -> list[Path]:
    """Yield direct children of each root that contain a config.yaml."""
    dirs = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for child in sorted(p.iterdir()):
            if child.is_dir() and (child / "config.yaml").exists():
                dirs.append(child)
    return dirs


def _patch_config_yaml(config_path: Path, old_folder: str, new_folder: str) -> None:
    """Update experiment_name_suffix and save fields in config.yaml in-place."""
    text = config_path.read_text()

    # Strip trailing _io from experiment_name_suffix (the canonical name no longer needs it
    # because fusion_branches is now embedded in the fusion tag).
    def _strip_trailing_fb(suffix: str) -> str:
        # Remove trailing _io, _im, _imo, _ip (fusion_branches tokens)
        cleaned = re.sub(r'_(?:io|im|imo|ip|iop|imop)$', '', suffix, flags=re.IGNORECASE)
        return cleaned

    # Replace experiment_name_suffix line
    def _replace_suffix(m: re.Match) -> str:
        key = m.group(1)
        val = m.group(2).strip()
        new_val = _strip_trailing_fb(val)
        if new_val == val:
            return m.group(0)
        return f"{key}{new_val}"

    text = re.sub(
        r'(experiment_name_suffix:\s*)(\S.*)',
        _replace_suffix,
        text,
    )

    # Replace save path (old_folder → new_folder)
    if old_folder in text:
        text = text.replace(old_folder, new_folder)

    config_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename experiment dirs to canonical names")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["output/drive", "output/chase", "output/octa500", "output/isic",
                 "output/deepglobe", "output/deepcrack", "output/cremi"],
        help="Dataset output roots to scan",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print renames without executing")
    args = parser.parse_args()

    experiment_dirs = _find_experiment_dirs(args.roots)
    renames: list[tuple[Path, Path]] = []

    for exp_dir in experiment_dirs:
        config = load_runtime_config(str(exp_dir))
        if config is None:
            continue
        canonical = build_experiment_name_from_runtime_config(config)
        if canonical is None:
            continue
        actual = exp_dir.name
        if actual == canonical:
            continue
        new_dir = exp_dir.parent / canonical
        renames.append((exp_dir, new_dir))

    if not renames:
        print("No renames needed — all folders already match canonical names.")
        return

    print(f"{'DRY RUN — ' if args.dry_run else ''}Found {len(renames)} folder(s) to rename:\n")
    for old, new in renames:
        print(f"  {old.parent.name}/{old.name}")
        print(f"  → {new.name}\n")

    if args.dry_run:
        print("Dry-run mode: no changes made.")
        return

    confirm = input(f"Rename {len(renames)} folder(s)? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    errors = 0
    for old, new in renames:
        if new.exists():
            print(f"[SKIP] target already exists: {new}")
            errors += 1
            continue
        config_path = old / "config.yaml"
        if config_path.exists():
            _patch_config_yaml(config_path, old.name, new.name)
        old.rename(new)
        print(f"[OK] {old.name} → {new.name}")

    if errors:
        print(f"\n{errors} folder(s) skipped due to conflicts.")
    else:
        print(f"\nRenamed {len(renames)} folder(s) successfully.")


if __name__ == "__main__":
    main()
