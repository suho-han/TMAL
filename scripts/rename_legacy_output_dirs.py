#!/usr/bin/env python3
"""
Rename legacy output experiment directories where fusion_branches was appended
as a suffix at the end instead of embedded right after the conn_fusion name.

Old: dist_inverted_dg_direct_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3_io
New: dist_inverted_dg_direct_io_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3

Usage:
    python scripts/rename_legacy_output_dirs.py output/          # dry-run
    python scripts/rename_legacy_output_dirs.py output/ --apply  # apply renames
"""

import argparse
import re
import sys
from pathlib import Path


KNOWN_FUSIONS = [
    "conv_residual",
    "scaled_sum",
    "decoder_guided",
    "dg_direct",
    "dg_dsf",
    "gate",
    "dsf",
    "dg",
]

# Ordered longest-first to avoid partial matches (iopf before iop, imo before im)
BRANCHES = ["iopf", "iop", "imo", "imp", "impf", "io", "ip", "im", "if", "i"]


def compute_new_name(old_name: str) -> tuple[str | None, str | None]:
    """Return (branch, new_name) or (branch, None) if no conn_fusion found."""
    branch = next((b for b in BRANCHES if old_name.endswith(f"_{b}")), None)
    if branch is None:
        return None, None

    body = old_name[: -(len(branch) + 1)]

    # Search for _{fusion}[_rs<X>][_<SINGLE_UPPER>]_<conn_num>_ in body.
    # We prepend/append _ so the regex can anchor on word boundaries reliably.
    padded = "_" + body + "_"
    for fusion in KNOWN_FUSIONS:
        pattern = (
            r"_"
            + re.escape(fusion)
            + r"(_rs[0-9][0-9.]*)?(_[A-Z])?(_[0-9]+)_"
        )
        m = re.search(pattern, padded)
        if m:
            # group(3) is the `_<conn_num>` token; insert branch before it.
            conn_num_start_in_padded = m.start(3)
            # Subtract 1 because we prepended one `_` to get padded.
            insert_pos = conn_num_start_in_padded - 1
            new_name = body[:insert_pos] + f"_{branch}" + body[insert_pos:]
            return branch, new_name

    return branch, None  # branch found but no conn_fusion to anchor on


def has_results(directory: Path) -> bool:
    return any(directory.glob("final_results_*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename legacy output dirs: move fusion_branches suffix to after conn_fusion.",
    )
    parser.add_argument("root", nargs="?", default="output",
                        help="Root output directory (default: output/)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually perform renames (default: dry-run only)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    to_rename: list[tuple[Path, Path]] = []
    conflicts: list[tuple[Path, Path]] = []
    skipped_no_fusion: list[Path] = []

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for exp_dir in sorted(dataset_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            branch, new_name = compute_new_name(exp_dir.name)
            if branch is None:
                continue
            if new_name is None:
                skipped_no_fusion.append(exp_dir)
                continue
            new_path = dataset_dir / new_name
            if new_path.exists():
                conflicts.append((exp_dir, new_path))
            else:
                to_rename.append((exp_dir, new_path))

    # ── Report ────────────────────────────────────────────────────────────────
    if to_rename:
        print(f"=== Proposed renames ({len(to_rename)}) ===")
        for old, new in to_rename:
            print(f"  {old.parent.name}/{old.name}")
            print(f"    → {new.name}")
        print()

    # Resolve conflicts where old is empty (no results) — safe to delete old.
    auto_deleted: list[Path] = []
    unresolved_conflicts: list[tuple[Path, Path]] = []
    for old, new in conflicts:
        if not has_results(old):
            auto_deleted.append(old)
        else:
            unresolved_conflicts.append((old, new))

    if unresolved_conflicts:
        print(f"=== Unresolved conflicts — target exists AND old has results ({len(unresolved_conflicts)}) ===")
        for old, new in unresolved_conflicts:
            new_has = has_results(new)
            status = f"new={'✓' if new_has else '✗'}"
            print(f"  {old.parent.name}/{old.name}")
            print(f"    → {new.name}  [{status}]")
        print()

    if auto_deleted:
        print(f"=== Auto-delete (old is empty, new already exists) ({len(auto_deleted)}) ===")
        for old in auto_deleted:
            print(f"  {old.parent.name}/{old.name}  [will be removed]")
        print()

    if skipped_no_fusion:
        print(f"=== Skipped — no conn_fusion found ({len(skipped_no_fusion)}) ===")
        for p in skipped_no_fusion:
            print(f"  {p.parent.name}/{p.name}")
        print()

    print(f"Summary: {len(to_rename)} to rename, {len(auto_deleted)} auto-delete, "
          f"{len(unresolved_conflicts)} unresolved conflicts, {len(skipped_no_fusion)} skipped.")

    if not args.apply:
        print("\nDry-run mode. Use --apply to perform renames.")
        return

    if unresolved_conflicts:
        print("\nERROR: resolve unresolved conflicts before applying renames.", file=sys.stderr)
        sys.exit(1)

    import shutil
    for old in auto_deleted:
        shutil.rmtree(old)
        print(f"Deleted empty: {old.parent.name}/{old.name}")

    for old, new in to_rename:
        old.rename(new)
        print(f"Renamed: {old.parent.name}/{old.name} → {new.name}")

    print(f"\nDone. Deleted {len(auto_deleted)}, renamed {len(to_rename)} directories.")


if __name__ == "__main__":
    main()
