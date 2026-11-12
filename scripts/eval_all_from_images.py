#!/usr/bin/env python3
"""
Batch-run eval_from_images.py across all experiments with checkpoint_batches.
Skips experiments that already have per-sample rows in final_results.csv.

Usage:
    uv run scripts/eval_all_from_images.py
    uv run scripts/eval_all_from_images.py --workers 16
    uv run scripts/eval_all_from_images.py --force   # re-run even if already done
"""

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make src importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from eval_from_images import (
    find_eval_epoch, read_final_results, read_existing_sample_csv,
    discover_sample_images, load_binary_png, compute_all_metrics,
    write_sample_metrics_csv, write_final_results_csv,
)


def already_done(exp_dir: str) -> bool:
    """True if final_results.csv already has per-sample rows."""
    path = os.path.join(exp_dir, 'final_results.csv')
    if not os.path.exists(path):
        return False
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        next(reader, None)          # skip aggregate row
        second = next(reader, None) # first sample row (if any)
    return second is not None and second.get('result_type') == 'sample'


def process_one(exp_dir: str) -> str:
    """Process a single experiment. Returns status string."""
    import math
    import numpy as np

    epoch = find_eval_epoch(exp_dir)
    if epoch is None:
        return f'SKIP_NO_EPOCH\t{exp_dir}'

    epoch_dir = os.path.join(exp_dir, 'models', 'checkpoint_batches', f'epoch_{epoch:03d}')
    if not os.path.isdir(epoch_dir):
        return f'SKIP_NO_EPOCH_DIR\t{exp_dir}'

    pred_paths, mask_paths, mode = discover_sample_images(epoch_dir)
    if not pred_paths or len(pred_paths) != len(mask_paths):
        return f'SKIP_NO_IMAGES\t{exp_dir}'

    existing_fields, existing_agg_row = read_final_results(exp_dir)
    existing_sample_rows = read_existing_sample_csv(exp_dir)

    sample_rows = []
    dice_ls, jac_ls, cldice_ls = [], [], []
    betti0_ls, betti1_ls, hd95_ls = [], [], []
    prec_ls, acc_ls = [], []

    for i, (pred_path, mask_path) in enumerate(zip(pred_paths, mask_paths)):
        pred_np = load_binary_png(pred_path)
        mask_np = load_binary_png(mask_path)
        m = compute_all_metrics(pred_np, mask_np)

        if mode == 'samples':
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

        sample_rows.append({
            'epoch':           existing.get('epoch', epoch),
            'batch':           existing.get('batch', i),
            'sample_in_batch': existing.get('sample_in_batch', 0),
            'sample_name':     sample_name,
            'val_loss':        existing.get('val_loss', float('nan')),
            'precision':       prec_val,
            'accuracy':        acc_val,
            **m,
        })

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

    write_sample_metrics_csv(exp_dir, sample_rows)
    write_final_results_csv(exp_dir, agg_metrics, sample_rows, existing_fields, existing_agg_row)
    return f'OK\t{exp_dir}\t{len(sample_rows)} samples\tdice={agg_metrics["dice"]:.4f}\thd95={agg_metrics["hd95"]:.4f}'


def process_safe(args):
    exp_dir, force = args
    try:
        if not force and already_done(exp_dir):
            return f'ALREADY_DONE\t{exp_dir}'
        return process_one(exp_dir)
    except Exception as e:
        return f'ERROR\t{exp_dir}\t{e}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=24,
                        help='Parallel workers (default: 24)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if already done')
    parser.add_argument('--root', default='output',
                        help='Root output directory (default: output)')
    args = parser.parse_args()

    # Discover all experiment directories
    exp_dirs = []
    for dirpath, dirnames, _ in os.walk(args.root):
        if os.path.basename(dirpath) == 'checkpoint_batches':
            exp_dir = dirpath.replace('/models/checkpoint_batches', '')
            exp_dirs.append(exp_dir)
    exp_dirs.sort()

    print(f'[INFO] Found {len(exp_dirs)} experiments with checkpoint_batches')
    already_count = sum(1 for d in exp_dirs if already_done(d))
    pending = len(exp_dirs) - already_count if not args.force else len(exp_dirs)
    print(f'[INFO] Already done: {already_count}  |  Pending: {pending}  |  Workers: {args.workers}')
    if pending == 0:
        print('[INFO] Nothing to do.')
        return
    print()

    tasks = [(d, args.force) for d in exp_dirs]
    counters = {'OK': 0, 'ALREADY_DONE': 0, 'SKIP_NO_EPOCH': 0,
                'SKIP_NO_EPOCH_DIR': 0, 'SKIP_NO_IMAGES': 0, 'ERROR': 0}
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_safe, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            status = result.split('\t')[0]
            counters[status] = counters.get(status, 0) + 1
            if status == 'ERROR':
                errors.append(result)
            if status not in ('ALREADY_DONE',):
                print(f'[{done:4d}/{len(tasks)}] {result}')
            elif done % 50 == 0:
                print(f'[{done:4d}/{len(tasks)}] ... ({counters["ALREADY_DONE"]} already done so far)')

    print()
    print('── Summary ──────────────────────────────────')
    for k, v in counters.items():
        if v:
            print(f'  {k:<20}: {v}')
    if errors:
        print('\nErrors:')
        for e in errors:
            print(f'  {e}')


if __name__ == '__main__':
    main()
