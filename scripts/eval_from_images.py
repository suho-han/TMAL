#!/usr/bin/env python3
"""
Compute per-sample metrics from saved checkpoint_batches pred/mask images
and write results to final_results.csv (per-sample rows) and
models/test_sample_metrics.csv.

Sources (checked in priority order):
  1. <epoch_dir>/pred_samples/<sample_name>.png  — per-sample (post-modification)
  2. <epoch_dir>/pred/batch_XXXX.png             — batch-level (batch_size=1 only)

Output:
  <exp_dir>/final_results.csv          — aggregate row (row 0, backward-compat) +
                                          per-sample rows (result_type=sample)
  <exp_dir>/models/test_sample_metrics.csv — per-sample metrics including hd95

Usage:
    uv run scripts/eval_from_images.py <exp_dir>
    uv run scripts/eval_from_images.py <exp_dir> --epoch 500
    uv run scripts/eval_from_images.py output/drive/dist_inverted_dg_direct_A_8_smooth_l1_segaux_w0.5
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

SAMPLE_CSV_FIELDS = [
    'epoch', 'batch', 'sample_in_batch', 'sample_name', 'val_loss',
    'dice', 'jac', 'cldice', 'precision', 'accuracy',
    'betti_error_0', 'betti_error_1', 'hd95',
]

FINAL_RESULTS_FIELDS = [
    'result_type', 'evaluated_split', 'eval_epoch', 'checkpoint',
    'train_loss', 'eval_loss',
    'dice', 'jac', 'cldice', 'precision', 'accuracy',
    'betti_error_0', 'betti_error_1', 'hd95', 'elapsed_hms',
    'dice_std', 'jac_std', 'cldice_std', 'precision_std', 'accuracy_std',
    'betti_error_0_std', 'betti_error_1_std', 'hd95_std',
]


# ── Image I/O ─────────────────────────────────────────────────────────────────

def load_binary_png(path: str) -> np.ndarray:
    """Load a PNG as a binary float32 array (threshold at 0.5 after [0,1] scaling)."""
    arr = np.array(Image.open(path).convert('L'), dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


# ── Metric computation ────────────────────────────────────────────────────────

def dice_jac(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    tp = float(np.sum(pred * gt))
    fp = float(np.sum(pred * (1.0 - gt)))
    fn = float(np.sum((1.0 - pred) * gt))
    denom_dice = 2 * tp + fp + fn
    denom_jac = tp + fp + fn
    d = (2 * tp / denom_dice) if denom_dice > 0 else float('nan')
    j = (tp / denom_jac) if denom_jac > 0 else float('nan')
    return d, j


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    d, j = dice_jac(pred, gt)
    cldice_val = float(clDice(pred, gt))
    betti0_errs, betti1_errs = getBettiErrors(pred, gt)
    betti0 = float(np.mean(betti0_errs)) if betti0_errs else float('nan')
    betti1 = float(np.mean(betti1_errs)) if betti1_errs else float('nan')
    hd_val = float(compute_hd95(pred, gt))
    return {
        'dice': d, 'jac': j, 'cldice': cldice_val,
        'betti_error_0': betti0, 'betti_error_1': betti1, 'hd95': hd_val,
    }


# ── CSV helpers ───────────────────────────────────────────────────────────────

def read_final_results(exp_dir: str) -> Tuple[Optional[List[str]], Optional[Dict]]:
    path = os.path.join(exp_dir, 'final_results.csv')
    if not os.path.exists(path):
        return None, None
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        first_row = next(reader, None)
    return fieldnames, first_row


def read_existing_sample_csv(exp_dir: str) -> List[Dict]:
    path = os.path.join(exp_dir, 'models', 'test_sample_metrics.csv')
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r.get('batch', 0)), int(r.get('sample_in_batch', 0))))
    return rows


def find_eval_epoch(exp_dir: str) -> Optional[int]:
    _, row = read_final_results(exp_dir)
    if row is None:
        return None
    raw = str(row.get('eval_epoch', '')).strip()
    return int(raw) if raw.isdigit() else None


# ── Image discovery ────────────────────────────────────────────────────────────

def discover_sample_images(epoch_dir: str) -> Tuple[List[str], List[str], str]:
    """Return sorted (pred_paths, mask_paths, mode) where mode is 'samples' or 'batch'."""
    pred_samples = os.path.join(epoch_dir, 'pred_samples')
    mask_samples = os.path.join(epoch_dir, 'mask_samples')
    if os.path.isdir(pred_samples) and os.path.isdir(mask_samples):
        pred_files = sorted(
            os.path.join(pred_samples, f)
            for f in os.listdir(pred_samples) if f.endswith('.png')
        )
        mask_files = sorted(
            os.path.join(mask_samples, f)
            for f in os.listdir(mask_samples) if f.endswith('.png')
        )
        return pred_files, mask_files, 'samples'

    pred_batch = os.path.join(epoch_dir, 'pred')
    mask_batch = os.path.join(epoch_dir, 'mask')
    if os.path.isdir(pred_batch) and os.path.isdir(mask_batch):
        pred_files = sorted(
            os.path.join(pred_batch, f)
            for f in os.listdir(pred_batch) if f.endswith('.png')
        )
        mask_files = sorted(
            os.path.join(mask_batch, f)
            for f in os.listdir(mask_batch) if f.endswith('.png')
        )
        return pred_files, mask_files, 'batch'

    return [], [], 'none'


# ── Write outputs ─────────────────────────────────────────────────────────────

def write_sample_metrics_csv(exp_dir: str, rows: List[Dict]) -> None:
    path = os.path.join(exp_dir, 'models', 'test_sample_metrics.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SAMPLE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_final_results_csv(
    exp_dir: str,
    agg_metrics: Dict[str, float],
    sample_rows: List[Dict],
    existing_fields: Optional[List[str]],
    existing_agg_row: Optional[Dict],
) -> None:
    path = os.path.join(exp_dir, 'final_results.csv')

    # Merge computed metrics into the existing aggregate row (or build a new one)
    if existing_agg_row is not None:
        agg_row = dict(existing_agg_row)
    else:
        agg_row = {f: '' for f in FINAL_RESULTS_FIELDS}
        agg_row['result_type'] = 'final'
        agg_row['evaluated_split'] = 'test'

    for k in ('dice', 'jac', 'cldice', 'betti_error_0', 'betti_error_1', 'hd95', 'precision', 'accuracy'):
        v = agg_metrics[k]
        agg_row[k] = f'{v:.6f}' if not math.isnan(v) else 'nan'
        # Also write STDs
        v_std = agg_metrics.get(f'{k}_std', float('nan'))
        agg_row[f'{k}_std'] = f'{v_std:.6f}' if not math.isnan(v_std) else 'nan'

    # Determine fieldnames (union of existing and standard)
    if existing_fields:
        fieldnames = list(existing_fields)
        for f in FINAL_RESULTS_FIELDS:
            if f not in fieldnames:
                fieldnames.append(f)
    else:
        fieldnames = FINAL_RESULTS_FIELDS

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        # Row 0: aggregate (backward-compat — parse_final_summary_csv reads only this row)
        writer.writerow(agg_row)
        # Rows 1..N: per-sample
        for sm in sample_rows:
            per_sample_row = {k: '' for k in fieldnames}
            per_sample_row['result_type'] = 'sample'
            per_sample_row['evaluated_split'] = str(sm['sample_name'])
            per_sample_row['eval_epoch'] = str(sm['epoch'])
            for k in ('dice', 'jac', 'cldice', 'betti_error_0', 'betti_error_1', 'hd95',
                      'precision', 'accuracy'):
                v = float(sm.get(k, float('nan')))
                per_sample_row[k] = f'{v:.6f}' if not math.isnan(v) else 'nan'
            writer.writerow(per_sample_row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('exp_dir', help='Experiment output directory')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Checkpoint epoch (auto-detected from final_results.csv if omitted)')
    args = parser.parse_args()

    exp_dir = args.exp_dir.rstrip('/')
    if not os.path.isdir(exp_dir):
        print(f'[ERROR] Not a directory: {exp_dir}')
        sys.exit(1)

    # Determine epoch
    epoch = args.epoch or find_eval_epoch(exp_dir)
    if epoch is None:
        print('[ERROR] Could not detect eval epoch. Use --epoch N.')
        sys.exit(1)
    print(f'[INFO] Experiment : {exp_dir}')
    print(f'[INFO] Eval epoch : {epoch}')

    # Find checkpoint images
    epoch_dir = os.path.join(exp_dir, 'models', 'checkpoint_batches', f'epoch_{epoch:03d}')
    if not os.path.isdir(epoch_dir):
        print(f'[ERROR] checkpoint_batches dir not found: {epoch_dir}')
        sys.exit(1)

    pred_paths, mask_paths, mode = discover_sample_images(epoch_dir)
    if not pred_paths:
        print(f'[ERROR] No pred/mask images found under {epoch_dir}')
        sys.exit(1)
    if len(pred_paths) != len(mask_paths):
        print(f'[ERROR] pred/mask count mismatch: {len(pred_paths)} vs {len(mask_paths)}')
        sys.exit(1)

    print(f'[INFO] Image mode : {mode} ({len(pred_paths)} images)')

    # Read existing metadata
    existing_fields, existing_agg_row = read_final_results(exp_dir)
    existing_sample_rows = read_existing_sample_csv(exp_dir)

    # Compute per-sample metrics
    print()
    print(f"{'Sample':<20} {'Dice':>7} {'Jac':>7} {'clDice':>7} {'Betti0':>7} {'Betti1':>7} {'HD95':>7}")
    print('-' * 65)

    sample_rows: List[Dict] = []
    dice_ls, jac_ls, cldice_ls = [], [], []
    betti0_ls, betti1_ls, hd95_ls = [], [], []
    prec_ls, acc_ls = [], []

    for i, (pred_path, mask_path) in enumerate(zip(pred_paths, mask_paths)):
        pred_np = load_binary_png(pred_path)
        mask_np = load_binary_png(mask_path)

        m = compute_all_metrics(pred_np, mask_np)

        # Metadata from existing CSV (if batch-mode and rows match up)
        if mode == 'samples':
            # sample name is the filename stem
            sample_name = os.path.splitext(os.path.basename(pred_path))[0]
            existing = next(
                (r for r in existing_sample_rows if str(r.get('sample_name', '')) == sample_name),
                existing_sample_rows[i] if i < len(existing_sample_rows) else {},
            )
        else:
            existing = existing_sample_rows[i] if i < len(existing_sample_rows) else {}
            sample_name = str(existing.get('sample_name', f'batch_{i:04d}'))

        dice_ls.append(m['dice']); jac_ls.append(m['jac']); cldice_ls.append(m['cldice'])
        betti0_ls.append(m['betti_error_0']); betti1_ls.append(m['betti_error_1'])
        hd95_ls.append(m['hd95'])

        # Collect precision/accuracy if available
        prec_val = float(existing.get('precision', float('nan')))
        acc_val = float(existing.get('accuracy', float('nan')))
        prec_ls.append(prec_val); acc_ls.append(acc_val)

        row = {
            'epoch': existing.get('epoch', epoch),
            'batch': existing.get('batch', i),
            'sample_in_batch': existing.get('sample_in_batch', 0),
            'sample_name': sample_name,
            'val_loss': existing.get('val_loss', float('nan')),
            'precision': existing.get('precision', float('nan')),
            'accuracy': existing.get('accuracy', float('nan')),
            **m,
        }
        sample_rows.append(row)

        def _f(v):
            return f'{v:7.4f}' if not math.isnan(v) else '    nan'
        print(f"{sample_name:<20} {_f(m['dice'])} {_f(m['jac'])} {_f(m['cldice'])} "
              f"{_f(m['betti_error_0'])} {_f(m['betti_error_1'])} {_f(m['hd95'])}")

    print('-' * 65)

    def _nanmean(lst):
        valid = [v for v in lst if not math.isnan(v)]
        return float(np.mean(valid)) if valid else float('nan')

    def _nanstd(lst):
        valid = [v for v in lst if not math.isnan(v)]
        return float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

    agg_metrics = {
        'dice':          _nanmean(dice_ls),
        'jac':           _nanmean(jac_ls),
        'cldice':        _nanmean(cldice_ls),
        'betti_error_0': _nanmean(betti0_ls),
        'betti_error_1': _nanmean(betti1_ls),
        'hd95':          _nanmean(hd95_ls),
        'precision':     _nanmean(prec_ls),
        'accuracy':      _nanmean(acc_ls),
        # STDs
        'dice_std':          _nanstd(dice_ls),
        'jac_std':           _nanstd(jac_ls),
        'cldice_std':        _nanstd(cldice_ls),
        'betti_error_0_std': _nanstd(betti0_ls),
        'betti_error_1_std': _nanstd(betti1_ls),
        'hd95_std':          _nanstd(hd95_ls),
        'precision_std':     _nanstd(prec_ls),
        'accuracy_std':      _nanstd(acc_ls),
    }

    def _fmt_ms(m, s):
        if math.isnan(m): return '    nan'
        return f'{m:.4f}±{s:.4f}'

    print(f"{'MEAN±STD':<20} "
          f"{_fmt_ms(agg_metrics['dice'], agg_metrics['dice_std'])} "
          f"{_fmt_ms(agg_metrics['jac'], agg_metrics['jac_std'])} "
          f"{_fmt_ms(agg_metrics['cldice'], agg_metrics['cldice_std'])} "
          f"{_fmt_ms(agg_metrics['betti_error_0'], agg_metrics['betti_error_0_std'])} "
          f"{_fmt_ms(agg_metrics['betti_error_1'], agg_metrics['betti_error_1_std'])} "
          f"{_fmt_ms(agg_metrics['hd95'], agg_metrics['hd95_std'])}")
    print()

    # Write outputs
    write_sample_metrics_csv(exp_dir, sample_rows)
    print(f"[INFO] Wrote: {exp_dir}/models/test_sample_metrics.csv")

    write_final_results_csv(exp_dir, agg_metrics, sample_rows, existing_fields, existing_agg_row)
    print(f"[INFO] Wrote: {exp_dir}/final_results.csv  (aggregate row + {len(sample_rows)} per-sample rows)")


if __name__ == '__main__':
    main()
