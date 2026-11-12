#!/usr/bin/env python3
"""
Compute per-sample metrics from an external model's prediction images and write
results to the experiment's final_results.csv (aggregate row + per-sample rows)
and models/test_sample_metrics.csv.

Used for models that saved predictions outside the standard checkpoint_batches
structure (e.g. diffusion models that dump per-image JPGs).

Usage:
    uv run scripts/eval_external_from_images.py \\
        --pred-dir /path/to/preds \\
        --pred-suffix _output_ens.jpg \\
        --gt-dir data/DeepCrack/test_lab \\
        --gt-ext .png \\
        --exp-dir output/baseline/deepcrack/medsegdiff \\
        --epoch 100000

Naming convention:
    pred file  → {sample_name}{pred_suffix}
    gt   file  → {sample_name}{gt_ext}

All image extensions supported by PIL are accepted for both pred and GT.
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics.cldice import clDice
from metrics.hd95 import hd95 as compute_hd95
from metrics.cal_betti import getBettiErrors

# Re-use CSV helpers from eval_from_images
sys.path.insert(0, os.path.dirname(__file__))
from eval_from_images import (
    SAMPLE_CSV_FIELDS,
    FINAL_RESULTS_FIELDS,
    read_final_results,
    write_sample_metrics_csv,
    write_final_results_csv,
)


def load_binary_image(path: str, threshold: float = 0.5) -> np.ndarray:
    arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return (arr > threshold).astype(np.float32)


def dice_jac(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    tp = float(np.sum(pred * gt))
    fp = float(np.sum(pred * (1.0 - gt)))
    fn = float(np.sum((1.0 - pred) * gt))
    denom_dice = 2 * tp + fp + fn
    denom_jac  = tp + fp + fn
    d = (2 * tp / denom_dice) if denom_dice > 0 else float('nan')
    j = (tp / denom_jac)      if denom_jac  > 0 else float('nan')
    return d, j


def _resize_to_match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Resize pred to gt's spatial dimensions if they differ."""
    if pred.shape == gt.shape:
        return pred
    h, w = gt.shape
    # pred is already binary float32; upsample with NEAREST and re-threshold at >0
    img = Image.fromarray((pred * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    return (np.array(img, dtype=np.float32) > 0).astype(np.float32)


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = _resize_to_match(pred, gt)
    d, j = dice_jac(pred, gt)
    cldice_val = float(clDice(pred, gt))
    b0_errs, b1_errs = getBettiErrors(pred, gt)
    b0 = float(np.mean(b0_errs)) if b0_errs else float('nan')
    b1 = float(np.mean(b1_errs)) if b1_errs else float('nan')
    hd = float(compute_hd95(pred, gt))
    return {'dice': d, 'jac': j, 'cldice': cldice_val,
            'betti_error_0': b0, 'betti_error_1': b1, 'hd95': hd}


def discover_pairs(pred_dir: str, pred_suffix: str,
                   gt_dir: str, gt_ext: str) -> List[Tuple[str, str, str]]:
    """Return sorted (sample_name, pred_path, gt_path) triples."""
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"pred_dir not found: {pred_dir}")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")

    pairs = []
    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(pred_suffix):
            continue
        sample_name = fname[: -len(pred_suffix)]
        gt_path = os.path.join(gt_dir, sample_name + gt_ext)
        if not os.path.isfile(gt_path):
            print(f"[WARN] GT not found for '{sample_name}': {gt_path}")
            continue
        pairs.append((sample_name, os.path.join(pred_dir, fname), gt_path))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred-dir',    required=True, help='Directory with prediction images')
    parser.add_argument('--pred-suffix', required=True,
                        help='Filename suffix to strip for sample name (e.g. _output_ens.jpg)')
    parser.add_argument('--gt-dir',      required=True, help='Directory with GT mask images')
    parser.add_argument('--gt-ext',      required=True,
                        help='GT filename extension (e.g. .png)')
    parser.add_argument('--exp-dir',     required=True,
                        help='Experiment output dir (where final_results.csv lives)')
    parser.add_argument('--epoch',          type=int, default=0,
                        help='Epoch number to record (cosmetic; default: read from final_results.csv)')
    parser.add_argument('--pred-threshold', type=float, default=0.5,
                        help='Binarization threshold for pred images in [0,1] (default: 0.5). '
                             'Use 0 for already-binarized images (any non-zero = foreground).')
    args = parser.parse_args()

    exp_dir     = args.exp_dir.rstrip('/')
    pred_dir    = args.pred_dir
    pred_suffix = args.pred_suffix
    gt_dir      = args.gt_dir
    gt_ext      = args.gt_ext if args.gt_ext.startswith('.') else '.' + args.gt_ext
    pred_thr    = args.pred_threshold

    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)

    # Determine epoch
    existing_fields, existing_agg_row = read_final_results(exp_dir)
    epoch = args.epoch
    if epoch == 0 and existing_agg_row is not None:
        raw = str(existing_agg_row.get('eval_epoch', '')).strip()
        if raw.isdigit():
            epoch = int(raw)

    # Discover pred/GT pairs
    pairs = discover_pairs(pred_dir, pred_suffix, gt_dir, gt_ext)
    if not pairs:
        print('[ERROR] No matching pred/GT pairs found.')
        sys.exit(1)

    print(f'[INFO] Experiment : {exp_dir}')
    print(f'[INFO] Eval epoch : {epoch}')
    print(f'[INFO] Pairs found: {len(pairs)}')
    print()
    print(f"{'Sample':<25} {'Dice':>7} {'Jac':>7} {'clDice':>7} "
          f"{'Betti0':>7} {'Betti1':>7} {'HD95':>7}")
    print('-' * 72)

    sample_rows: List[Dict] = []
    lists = {k: [] for k in ('dice', 'jac', 'cldice', 'betti_error_0', 'betti_error_1', 'hd95', 'precision', 'accuracy')}

    for i, (sample_name, pred_path, gt_path) in enumerate(pairs):
        pred_np = load_binary_image(pred_path, threshold=pred_thr)
        gt_np   = load_binary_image(gt_path)
        m = compute_all_metrics(pred_np, gt_np)

        m['precision'] = float('nan')
        m['accuracy'] = float('nan')

        for k in lists:
            if not math.isnan(m[k]):
                lists[k].append(m[k])

        sample_rows.append({
            'epoch':           epoch,
            'batch':           i,
            'sample_in_batch': 0,
            'sample_name':     sample_name,
            'val_loss':        float('nan'),
            'precision':       float('nan'),
            'accuracy':        float('nan'),
            **m,
        })

        def _f(v):
            return f'{v:7.4f}' if not math.isnan(v) else '    nan'
        print(f"{sample_name:<25} {_f(m['dice'])} {_f(m['jac'])} {_f(m['cldice'])} "
              f"{_f(m['betti_error_0'])} {_f(m['betti_error_1'])} {_f(m['hd95'])}")

    print('-' * 72)

    def _nanmean(lst):
        return float(np.mean(lst)) if lst else float('nan')

    def _nanstd(lst):
        return float(np.std(lst, ddof=1)) if len(lst) > 1 else 0.0

    agg = {k: _nanmean(lists[k]) for k in lists}
    # Add STDs for compatibility with write_final_results_csv
    for k in list(lists.keys()):
        agg[f'{k}_std'] = _nanstd(lists[k])

    def _fmt_ms(m, s):
        if math.isnan(m): return '    nan'
        return f'{m:.4f}±{s:.4f}'

    print(f"{'MEAN±STD':<25} "
          f"{_fmt_ms(agg['dice'], agg['dice_std'])} "
          f"{_fmt_ms(agg['jac'], agg['jac_std'])} "
          f"{_fmt_ms(agg['cldice'], agg['cldice_std'])} "
          f"{_fmt_ms(agg['betti_error_0'], agg['betti_error_0_std'])} "
          f"{_fmt_ms(agg['betti_error_1'], agg['betti_error_1_std'])} "
          f"{agg['hd95']:7.4f}")  # hd95 might not have std in this script's print format
    print()

    write_sample_metrics_csv(exp_dir, sample_rows)
    print(f"[INFO] Wrote: {exp_dir}/models/test_sample_metrics.csv")

    write_final_results_csv(exp_dir, agg, sample_rows, existing_fields, existing_agg_row)
    print(f"[INFO] Wrote: {exp_dir}/final_results.csv  "
          f"(aggregate row + {len(sample_rows)} per-sample rows)")


if __name__ == '__main__':
    main()
