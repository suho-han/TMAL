#!/usr/bin/env python3
"""Build per-image normalized distance maps (and their inverted variant) from binary masks.

Mirrors the logic in ``notebooks/distance_map_drive.ipynb`` but works on any
directory of binary masks, not just DRIVE:

- binarize each mask (pixel > 0 is foreground)
- Euclidean distance transform (``distance_transform_edt``) inside the foreground
- min-max normalize to [0, 1] -> "dist" map
- invert the raw EDT within the foreground (``max - d``) before normalizing -> "dist_inverted" map

Known caveat (see ``build_dist_inverted_global.py``): the deepest foreground
pixel(s) attain ``dist_raw.max()``, so ``max_val - dist_raw`` sends them to
exactly 0 -- the same value as background. Pass ``--avoid_zero_collapse`` to
add a +1 offset so foreground never touches 0 (mirrors the "nz" fix used for
DRIVE). Default behaviour matches the notebook exactly (no offset).

Usage:
    python scripts/build_dist_maps.py --mask_dir data/FIVES/train/masks
    python scripts/build_dist_maps.py \
        --mask_dir data/DeepCrack/train_lab --mask_glob "*.png"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def normalize_01(x: np.ndarray) -> np.ndarray:
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max == x_min:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def build_distance_maps(binary: np.ndarray, avoid_zero_collapse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """binary: {0,1} array. Returns (dist, dist_inverted), both normalized to [0, 1]."""
    dist_raw = distance_transform_edt(binary)
    dist = normalize_01(dist_raw)

    dist_inv_raw = dist_raw.copy()
    fg = dist_inv_raw > 0
    if np.any(fg):
        offset = 1.0 if avoid_zero_collapse else 0.0
        max_val = dist_inv_raw.max()
        dist_inv_raw[fg] = (max_val + offset) - dist_inv_raw[fg]
    dist_inverted = normalize_01(dist_inv_raw)

    return dist, dist_inverted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask_dir", type=Path, required=True, help="directory of binary mask files")
    p.add_argument("--mask_glob", type=str, default="*.png", help="glob pattern for mask files (e.g. '*.gif', '*.png')")
    p.add_argument("--out_dist_dir", type=Path, default=None,
                   help="output dir for dist maps (default: <mask_dir>_dist, sibling of mask_dir)")
    p.add_argument("--out_dist_inverted_dir", type=Path, default=None,
                   help="output dir for inverted dist maps (default: <mask_dir>_dist_inverted, sibling of mask_dir)")
    p.add_argument("--dist_suffix", type=str, default="_dist", help="filename suffix (before extension) for dist outputs")
    p.add_argument("--dist_inverted_suffix", type=str, default="_dist_inverted", help="filename suffix for dist_inverted outputs")
    p.add_argument("--avoid_zero_collapse", action="store_true",
                   help="offset the inverted map by +1 so the deepest foreground pixel doesn't collapse to 0")
    p.add_argument("--no_preview", action="store_true", help="skip the uint8 preview PNGs")
    p.add_argument("--force", action="store_true", help="overwrite existing output files instead of skipping them")
    args = p.parse_args()

    if args.out_dist_dir is None:
        args.out_dist_dir = args.mask_dir.parent / f"{args.mask_dir.name}_dist"
    if args.out_dist_inverted_dir is None:
        args.out_dist_inverted_dir = args.mask_dir.parent / f"{args.mask_dir.name}_dist_inverted"
    return args


def main() -> int:
    args = parse_args()

    mask_files = sorted(args.mask_dir.glob(args.mask_glob))
    if not mask_files:
        raise SystemExit(f"No masks matched {args.mask_glob!r} under {args.mask_dir}")

    args.out_dist_dir.mkdir(parents=True, exist_ok=True)
    args.out_dist_inverted_dir.mkdir(parents=True, exist_ok=True)

    print(f"mask_dir={args.mask_dir} ({len(mask_files)} files)")
    print(f"out_dist_dir={args.out_dist_dir}")
    print(f"out_dist_inverted_dir={args.out_dist_inverted_dir}")

    dist_sum, dist_count = 0.0, 0
    inv_sum, inv_count = 0.0, 0
    n_written, n_skipped = 0, 0

    for mask_path in mask_files:
        dist_npy = args.out_dist_dir / f"{mask_path.stem}{args.dist_suffix}.npy"
        inv_npy = args.out_dist_inverted_dir / f"{mask_path.stem}{args.dist_inverted_suffix}.npy"

        if not args.force and dist_npy.exists() and inv_npy.exists():
            n_skipped += 1
            continue

        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask > 0).astype(np.uint8)

        dist, dist_inverted = build_distance_maps(binary, avoid_zero_collapse=args.avoid_zero_collapse)

        np.save(dist_npy, dist)
        np.save(inv_npy, dist_inverted)

        if not args.no_preview:
            Image.fromarray((dist * 255.0).astype(np.uint8)).save(dist_npy.with_suffix(".png"))
            Image.fromarray((dist_inverted * 255.0).astype(np.uint8)).save(inv_npy.with_suffix(".png"))

        dist_sum += float(dist.sum())
        dist_count += dist.size
        inv_sum += float(dist_inverted.sum())
        inv_count += dist_inverted.size
        n_written += 1

    if dist_count:
        print(f"mean(dist)={dist_sum / dist_count:.6f}  mean(dist_inverted)={inv_sum / inv_count:.6f}")
    print(f"written={n_written} skipped={n_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
