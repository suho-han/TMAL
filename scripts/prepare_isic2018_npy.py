#!/usr/bin/env python3
"""Prepare resized ISIC2018 `.npy` artifacts from the raw image split tree."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

try:
    RESAMPLING = Image.Resampling
except AttributeError:  # Pillow < 9.1
    RESAMPLING = Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move the raw ISIC2018 split tree to data/ISIC2018_img and build "
            "a resized flat `.npy` dataset under data/ISIC2018."
        ),
    )
    parser.add_argument(
        "--migrate-from-root",
        default="data/ISIC2018",
        help=(
            "Legacy location that may still contain the raw split tree. "
            "If it has train/val/test images+labels and --raw-root is absent, "
            "it is renamed to --raw-root."
        ),
    )
    parser.add_argument(
        "--raw-root",
        default="data/ISIC2018_img",
        help="Location of the raw split tree with train/val/test images+labels.",
    )
    parser.add_argument(
        "--output-root",
        default="data/ISIC2018",
        help="Output location for flat `.npy` artifacts.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Resized output height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Resized output width.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild `.npy` outputs even when matching files already exist.",
    )
    return parser.parse_args()


def has_raw_split_tree(root: Path) -> bool:
    return all(
        (root / split / "images").is_dir() and (root / split / "labels").is_dir()
        for split in SPLITS
    )


def ensure_raw_root(migrate_from_root: Path, raw_root: Path) -> None:
    if raw_root.exists():
        if not has_raw_split_tree(raw_root):
            raise RuntimeError(
                f"{raw_root} exists but does not look like a raw ISIC2018 split tree."
            )
        return

    if migrate_from_root.exists() and has_raw_split_tree(migrate_from_root):
        raw_root.parent.mkdir(parents=True, exist_ok=True)
        migrate_from_root.rename(raw_root)
        print(f"Moved raw ISIC2018 tree: {migrate_from_root} -> {raw_root}")
        return

    raise FileNotFoundError(
        "Raw ISIC2018 split tree not found. Expected either:\n"
        f"  - {raw_root}\n"
        f"  - {migrate_from_root} (to migrate into {raw_root})"
    )


def collect_pairs(raw_root: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for split in SPLITS:
        image_dir = raw_root / split / "images"
        label_dir = raw_root / split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = image_path.stem
            label_path = label_dir / f"{stem}_segmentation.png"
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label for {image_path}: expected {label_path}"
                )
            pairs.append((split, image_path, label_path))
    return pairs


def output_pair_matches(
    image_out_path: Path,
    label_out_path: Path,
    height: int,
    width: int,
) -> bool:
    if not image_out_path.exists() or not label_out_path.exists():
        return False

    try:
        image_shape = np.load(image_out_path, mmap_mode="r").shape
        label_shape = np.load(label_out_path, mmap_mode="r").shape
    except Exception:
        return False

    return image_shape == (height, width, 3) and label_shape == (height, width)


def save_npy_pair(
    image_path: Path,
    label_path: Path,
    image_out_dir: Path,
    label_out_dir: Path,
    height: int,
    width: int,
    force: bool = False,
) -> bool:
    stem = image_path.stem
    image_out_path = image_out_dir / f"{stem}.npy"
    label_out_path = label_out_dir / f"{stem}_segmentation.npy"

    if not force and output_pair_matches(
        image_out_path=image_out_path,
        label_out_path=label_out_path,
        height=height,
        width=width,
    ):
        return False

    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
        image = image.resize((width, height), RESAMPLING.BILINEAR)
        image_np = np.asarray(image, dtype=np.uint8)

    with Image.open(label_path) as label_handle:
        label = label_handle.convert("L")
        label = label.resize((width, height), RESAMPLING.NEAREST)
        label_np = np.asarray(label, dtype=np.uint8)

    np.save(image_out_path, image_np)
    np.save(label_out_path, label_np)
    return True


def main() -> int:
    args = parse_args()

    migrate_from_root = Path(args.migrate_from_root)
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    image_out_dir = output_root / "image"
    label_out_dir = output_root / "label"

    ensure_raw_root(migrate_from_root, raw_root)

    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(raw_root)
    print(
        f"Preparing {len(pairs)} ISIC2018 samples "
        f"at {args.height}x{args.width} into {output_root}"
        f" (force={args.force})"
    )

    converted_count = 0
    skipped_count = 0
    for index, (split, image_path, label_path) in enumerate(pairs, start=1):
        converted = save_npy_pair(
            image_path=image_path,
            label_path=label_path,
            image_out_dir=image_out_dir,
            label_out_dir=label_out_dir,
            height=args.height,
            width=args.width,
            force=args.force,
        )
        if converted:
            converted_count += 1
        else:
            skipped_count += 1
        if index == 1 or index % 250 == 0 or index == len(pairs):
            print(
                f"[{index:04d}/{len(pairs):04d}] "
                f"{'converted' if converted else 'skipped'} {split} "
                f"{image_path.name} -> {image_path.stem}.npy"
            )

    image_count = len(list(image_out_dir.glob("*.npy")))
    label_count = len(list(label_out_dir.glob("*_segmentation.npy")))
    print(
        "Done. Output counts: "
        f"image={image_count}, label={label_count}, raw_root={raw_root}, "
        f"converted={converted_count}, skipped={skipped_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
