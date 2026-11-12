#!/usr/bin/env python3
"""Build zero-free inverted distance labels for DRIVE into a parallel data root.

The label shipped in ``data/DRIVE/*/masks_dist_inverted`` is built as::

    max_val = dist_raw.max()
    inv[dist_raw > 0] = max_val - dist_raw[dist_raw > 0]

``max_val`` is attained inside the mask, so the deepest foreground pixel(s)
collapse to exactly 0 and become indistinguishable from background. That matters
because 0 is read as background in three places: ``dist_target_to_mask`` (edge +
dice loss), ``dist_to_binary`` (evaluation GT), and the affinity target
``Ck = D * D_k * exp(-|D - D_k| / sigma)``, which is multiplicative in ``D`` and
therefore supervises those pixels as fully disconnected.

Two corrected variants are written, each into its own data root so that nothing
under ``data/DRIVE`` is touched:

``gn``  offset + dataset-global normalization::

            inv[fg] = (GMAX + 1 - dist_raw[fg]) / GMAX

        with ``GMAX`` the max EDT over the *train* split only (no test leakage).
        Foreground lands in ``[1/GMAX, 1.0]``, background stays 0, and vessel
        thickness is comparable across images.

``nz``  offset only, keeping the original per-image normalization::

            inv[fg] = (max_val + 1 - dist_raw[fg]) / max_val

        Isolates the zero fix from the normalization change.

Each root mirrors DRIVE's layout, symlinking ``images``/``masks`` back to the
originals and writing only ``masks_dist_inverted/``, which is the directory name
``MyDataset_DRIVE`` already looks for under whatever ``train_root`` it is given.
So the existing ``--label_mode dist_inverted`` path picks these up unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = REPO_ROOT / "data" / "DRIVE"
SPLITS = ("train", "test")
VARIANTS = ("gn", "nz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source_root", type=Path, default=SOURCE_ROOT,
                   help="read-only DRIVE root (default: data/DRIVE)")
    p.add_argument("--out_root_gn", type=Path, default=REPO_ROOT / "data" / "DRIVE_distgn",
                   help="output root for the offset + global-normalization variant")
    p.add_argument("--out_root_nz", type=Path, default=REPO_ROOT / "data" / "DRIVE_distnz",
                   help="output root for the offset-only variant")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--force", action="store_true",
                   help="replace an existing output root instead of refusing")
    p.add_argument("--no_preview", action="store_true", help="skip the uint8 preview pngs")
    return p.parse_args()


def mask_files(source_root: Path, split: str) -> list[Path]:
    files = sorted((source_root / split / "masks").glob("*.gif"))
    if not files:
        raise SystemExit(f"no masks found under {source_root / split / 'masks'}")
    return files


def edt_of(mask_path: Path) -> np.ndarray:
    """EDT of the binarized mask. Matches the notebook: threshold at > 0."""
    mask = np.array(Image.open(mask_path).convert("L"))
    return distance_transform_edt((mask > 0).astype(np.uint8))


def compute_gmax(source_root: Path) -> float:
    """Global max EDT over the train split only, so test stats never leak."""
    return max(float(edt_of(p).max()) for p in mask_files(source_root, "train"))


def invert(dist_raw: np.ndarray, denom: float) -> np.ndarray:
    """(denom + 1 - d) / denom on foreground, exact 0 on background.

    Boundary pixels (d == 1) map to 1.0 and the deepest pixel maps to
    1/denom > 0, so no foreground pixel collides with background.
    """
    fg = dist_raw > 0
    inv = np.zeros(dist_raw.shape, dtype=np.float32)
    inv[fg] = (denom + 1.0 - dist_raw[fg]) / denom
    return inv


def guard_output_root(out_root: Path, source_root: Path, force: bool) -> None:
    if out_root.resolve() == source_root.resolve() or source_root.resolve() in out_root.resolve().parents:
        raise SystemExit(f"refusing to write inside the source root: {out_root}")
    if out_root.exists():
        if not force:
            raise SystemExit(f"{out_root} already exists; pass --force to replace it")
        shutil.rmtree(out_root)


def link_shared_dirs(out_split: Path, source_root: Path, split: str) -> None:
    """Symlink images/ and masks/ back to the originals (read-only use)."""
    for name in ("images", "masks"):
        target = (source_root / split / name).resolve()
        link = out_split / name
        link.symlink_to(target, target_is_directory=True)


def build_variant(variant: str, out_root: Path, source_root: Path,
                  gmax: float, preview: bool) -> dict:
    per_image_max: dict[str, float] = {}
    fg_counts: dict[str, int] = {}

    for split in SPLITS:
        out_split = out_root / split
        label_dir = out_split / "masks_dist_inverted"
        label_dir.mkdir(parents=True)
        link_shared_dirs(out_split, source_root, split)

        for mask_path in mask_files(source_root, split):
            dist_raw = edt_of(mask_path)
            denom = gmax if variant == "gn" else float(dist_raw.max())
            inv = invert(dist_raw, denom)

            stem = mask_path.stem
            per_image_max[f"{split}/{stem}"] = float(dist_raw.max())
            fg_counts[f"{split}/{stem}"] = int((dist_raw > 0).sum())

            np.save(label_dir / f"{stem}_dist_inverted.npy", inv)
            if preview:
                Image.fromarray((inv * 255.0).astype(np.uint8)).save(
                    label_dir / f"{stem}_dist_inverted.png"
                )

    meta = {
        "variant": variant,
        "formula": ("(GMAX + 1 - d) / GMAX" if variant == "gn"
                    else "(img_max + 1 - d) / img_max"),
        "gmax": gmax if variant == "gn" else None,
        "gmax_source_split": "train" if variant == "gn" else None,
        "normalization": "dataset-global" if variant == "gn" else "per-image",
        "source_root": str(source_root),
        "n_images": len(per_image_max),
        "per_image_raw_max": per_image_max,
        "per_image_fg_count": fg_counts,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def verify(out_root: Path, source_root: Path, variant: str, gmax: float) -> None:
    """Assert the property the whole change exists for: no foreground zero."""
    checked = 0
    for split in SPLITS:
        label_dir = out_root / split / "masks_dist_inverted"
        for mask_path in mask_files(source_root, split):
            dist_raw = edt_of(mask_path)
            inv = np.load(label_dir / f"{mask_path.stem}_dist_inverted.npy")

            fg = dist_raw > 0
            floor = 1.0 / (gmax if variant == "gn" else float(dist_raw.max()))

            assert inv.shape == dist_raw.shape, f"{mask_path}: shape mismatch"
            assert inv.dtype == np.float32, f"{mask_path}: dtype {inv.dtype}"
            assert inv.max() <= 1.0 + 1e-6, f"{mask_path}: max {inv.max()} > 1"
            assert float(inv[~fg].max(initial=0.0)) == 0.0, f"{mask_path}: background not 0"
            assert float(inv[fg].min()) >= floor - 1e-6, (
                f"{mask_path}: foreground min {inv[fg].min()} below floor {floor}"
            )
            assert int((inv > 0).sum()) == int(fg.sum()), (
                f"{mask_path}: foreground count {(inv > 0).sum()} != {fg.sum()}"
            )
            checked += 1
    print(f"  verified {checked} files: no foreground pixel is 0, counts preserved")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source root not found: {source_root}")

    gmax = compute_gmax(source_root)
    print(f"GMAX (train split only) = {gmax:.4f}")

    roots = {"gn": args.out_root_gn, "nz": args.out_root_nz}
    for variant in args.variants:
        out_root = roots[variant].resolve()
        guard_output_root(out_root, source_root, args.force)
        print(f"\n[{variant}] -> {out_root}")
        meta = build_variant(variant, out_root, source_root, gmax, not args.no_preview)
        print(f"  wrote {meta['n_images']} labels  ({meta['formula']})")
        verify(out_root, source_root, variant, gmax)

    print(f"\nsource root untouched: {source_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
