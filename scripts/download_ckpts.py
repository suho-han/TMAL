#!/usr/bin/env python3
"""Download released TMAL checkpoints from the Hugging Face Hub."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download


DATASETS = ("drive", "deepcrack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--dest", type=Path, default=Path("ckpts"))
    parser.add_argument("--repo-id", default="suho-han/TMAL")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--file-format", choices=("pth", "safetensors"), default="pth")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)

    for dataset in datasets:
        if args.file_format == "pth":
            filename = f"{dataset}/{dataset}_model.pth"
            destination = args.dest / f"{dataset}_model.pth"
        else:
            filename = f"{dataset}/model.safetensors"
            destination = args.dest / f"{dataset}_model.safetensors"

        print(
            f"dataset={dataset} repo_id={args.repo_id} revision={args.revision} "
            f"destination={destination.resolve()}"
        )
        if destination.exists() and not args.force:
            print(f"skip {destination} (already exists)")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{dataset}-", dir=destination.parent
            ) as tmp_dir:
                downloaded = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=filename,
                    revision=args.revision,
                    local_dir=tmp_dir,
                )
                os.replace(downloaded, destination)
        except Exception as exc:
            print(f"error: could not download {filename}: {exc}", file=sys.stderr)
            print(
                "hint: the model repo may be private (set HF_TOKEN or run "
                "huggingface-cli login), or the network may be unavailable.",
                file=sys.stderr,
            )
            return 1

        print(f"downloaded {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
