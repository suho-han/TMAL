#!/usr/bin/env python3
"""Parameter-count comparison: Proposed / DconnNet vs. the in-repo baselines.

All curvilinear baselines are counted in the same task setup as the paper
comparison table (conn_num=8 connectivity output, num_class=1, 3-channel input),
matching how they are trained (e.g. scripts/configs/drive_csnet_train.yaml).

External baselines with no architecture in this repo (CAPE, V-GAN) cannot be
counted here; clDice is a loss (its backbone params equal the U-Net row).

Usage:
    python scripts/measure_baseline_params.py
    python scripts/measure_baseline_params.py --with-diffusion   # add SegDiff/MedSegDiff
"""
import argparse
import sys
from types import SimpleNamespace

sys.path.insert(0, 'src')

import torch  # noqa: F401  (registry build fns may touch torch/cuda)
from model.registry import build_model
from model.DconnNet import DconnNet

COMMON = dict(num_class=1, conn_num=8, conn_layout='standard8', fusion_residual_scale=1.0)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def make_args(**over):
    base = dict(num_channels=3, num_class=1, conn_num=8, conn_layout='standard8',
                simple_loss=False, image_size=512, diff_timesteps=100)
    base.update(over)
    return SimpleNamespace(**base)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--with-diffusion', action='store_true',
                    help='also count SegDiff / MedSegDiff (different paradigm, much larger)')
    args = ap.parse_args()

    rows = []  # (display, params)

    # DconnNet family (direct construction to control the exact fusion kwargs).
    dconn = DconnNet(**COMMON, conn_fusion='none', use_seg_aux=False).eval()
    rows.append(("DconnNet (baseline)", n_params(dconn)))

    baselines = [
        ("U-Net", "unet"),
        ("Attention U-Net", "attention_unet"),
        ("CSNet", "csnet"),
        ("FRU-Net", "frunet"),
    ]
    if args.with_diffusion:
        baselines += [("SegDiff", "segdiff"), ("MedSegDiff", "medsegdiff")]

    for disp, name in baselines:
        try:
            m = build_model(name, make_args()).eval()
            rows.append((disp, n_params(m)))
        except Exception as e:
            rows.append((disp, None))
            print(f"[skip] {disp} ({name}): {type(e).__name__}: {e}")

    prop_full = DconnNet(**COMMON, conn_fusion='dg_dsf', fusion_branches='io', use_seg_aux=True).eval()
    rows.append(("Proposed", n_params(prop_full)))

    print("\nParameter comparison (num_class=1, conn_num=8, 3-ch input)\n")
    hdr = f"{'Model':<24}{'Params(M)':>12}"
    print(hdr)
    print('-' * len(hdr))
    for disp, p in rows:
        if p is None:
            print(f"{disp:<24}{'N/A':>12}")
            continue
        print(f"{disp:<24}{p/1e6:>12.2f}")

    print("\nNot countable from this repo: CAPE, V-GAN (no architecture in src/model);")
    print("clDice is a loss — its backbone params equal the U-Net row.")


if __name__ == '__main__':
    main()
