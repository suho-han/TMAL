#!/usr/bin/env python3
"""Compute Wilcoxon signed-rank p-values between two experiment runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon


_METRIC_ALIASES = {
    "iou": "jac",
    "jac": "jac",
    "dice": "dice",
    "cldice": "cldice",
    "precision": "precision",
    "accuracy": "accuracy",
    "betti_error_0": "betti_error_0",
    "betti_error_1": "betti_error_1",
    "hd95": "hd95",
}


def _to_float(value: object) -> float:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return float("nan")
    return float(text)


def _normalize_metric(metric: str) -> str:
    key = metric.strip().lower()
    if key not in _METRIC_ALIASES:
        raise ValueError(f"Unsupported metric: {metric}")
    return _METRIC_ALIASES[key]


def _find_final_results_csv(exp_dir: str) -> Optional[str]:
    candidates = ["final_results.csv", "final_results_1.csv"]
    for name in candidates:
        path = os.path.join(exp_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _find_test_sample_csv(exp_dir: str) -> Optional[str]:
    primary = os.path.join(exp_dir, "models", "test_sample_metrics.csv")
    if os.path.isfile(primary):
        return primary
    models_dir = os.path.join(exp_dir, "models")
    if not os.path.isdir(models_dir):
        return None
    for entry in sorted(os.listdir(models_dir)):
        candidate = os.path.join(models_dir, entry, "test_sample_metrics.csv")
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_samples_from_final_results(path: str, metric: str) -> Dict[str, float]:
    samples: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("result_type", "")).strip().lower() != "sample":
                continue
            sample_name = str(row.get("evaluated_split", "")).strip()
            if not sample_name:
                sample_name = str(row.get("sample_name", "")).strip()
            if not sample_name:
                continue
            value = _to_float(row.get(metric, "nan"))
            if sample_name not in samples:
                samples[sample_name] = value
    return samples


def _load_samples_from_test_metrics(path: str, metric: str) -> Dict[str, float]:
    samples: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_name = str(row.get("sample_name", "")).strip()
            if not sample_name:
                continue
            value = _to_float(row.get(metric, "nan"))
            if sample_name not in samples:
                samples[sample_name] = value
    return samples


def load_samples(exp_dir: str, metric: str, source: str) -> Tuple[Dict[str, float], str]:
    if source == "final_results":
        path = _find_final_results_csv(exp_dir)
        if not path:
            raise FileNotFoundError(f"No final_results.csv found under {exp_dir}")
        return _load_samples_from_final_results(path, metric), path
    if source == "test_sample_metrics":
        path = _find_test_sample_csv(exp_dir)
        if not path:
            raise FileNotFoundError(f"No test_sample_metrics.csv found under {exp_dir}/models")
        return _load_samples_from_test_metrics(path, metric), path

    final_path = _find_final_results_csv(exp_dir)
    if final_path:
        samples = _load_samples_from_final_results(final_path, metric)
        if samples:
            return samples, final_path
    test_path = _find_test_sample_csv(exp_dir)
    if test_path:
        samples = _load_samples_from_test_metrics(test_path, metric)
        if samples:
            return samples, test_path
    raise FileNotFoundError(
        f"No per-sample metrics found under {exp_dir} "
        "(expected final_results.csv or models/test_sample_metrics.csv)."
    )


def _pair_samples(
    proposed: Dict[str, float],
    baseline: Dict[str, float],
) -> Tuple[List[str], List[float], List[float], List[str]]:
    shared = sorted(set(proposed) & set(baseline))
    used_names: List[str] = []
    prop_vals: List[float] = []
    base_vals: List[float] = []
    dropped: List[str] = []
    for name in shared:
        p_val = proposed[name]
        b_val = baseline[name]
        if math.isnan(float(p_val)) or math.isnan(float(b_val)):
            dropped.append(name)
            continue
        used_names.append(name)
        prop_vals.append(float(p_val))
        base_vals.append(float(b_val))
    return used_names, prop_vals, base_vals, dropped


def _mean_median(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.median(arr))


def _write_output(path: str, payload: Dict[str, object]) -> None:
    if path.lower().endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Wilcoxon signed-rank p-values between two experiment runs."
    )
    parser.add_argument("--proposed", required=True, help="Proposed experiment directory")
    parser.add_argument("--baseline", required=True, help="Baseline (DconnNet) experiment directory")
    parser.add_argument("--metric", default="dice", help="Metric key (default: dice)")
    parser.add_argument(
        "--alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help="Alternative hypothesis",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "final_results", "test_sample_metrics"),
        default="auto",
        help="Per-sample metric source",
    )
    parser.add_argument("--output", default="", help="Optional output CSV/JSON path")
    args = parser.parse_args()

    metric = _normalize_metric(args.metric)

    proposed_samples, proposed_source = load_samples(args.proposed, metric, args.source)
    baseline_samples, baseline_source = load_samples(args.baseline, metric, args.source)

    used_names, proposed_vals, baseline_vals, dropped = _pair_samples(
        proposed_samples, baseline_samples
    )

    missing_proposed = sorted(set(baseline_samples) - set(proposed_samples))
    missing_baseline = sorted(set(proposed_samples) - set(baseline_samples))

    if not proposed_vals:
        print("[ERROR] No valid paired samples after filtering NaNs.", file=sys.stderr)
        sys.exit(2)

    try:
        stat, p_value = wilcoxon(
            proposed_vals,
            baseline_vals,
            alternative=args.alternative,
        )
    except ValueError as exc:
        print(f"[ERROR] Wilcoxon failed: {exc}", file=sys.stderr)
        sys.exit(2)

    prop_mean, prop_median = _mean_median(proposed_vals)
    base_mean, base_median = _mean_median(baseline_vals)

    summary = {
        "metric": metric,
        "alternative": args.alternative,
        "n_total_shared": len(set(proposed_samples) & set(baseline_samples)),
        "n_used": len(proposed_vals),
        "n_dropped_nan": len(dropped),
        "statistic": float(stat),
        "p_value": float(p_value),
        "proposed_mean": prop_mean,
        "proposed_median": prop_median,
        "baseline_mean": base_mean,
        "baseline_median": base_median,
        "proposed_dir": os.path.abspath(args.proposed),
        "baseline_dir": os.path.abspath(args.baseline),
        "proposed_source": proposed_source,
        "baseline_source": baseline_source,
    }

    print(f"[INFO] Metric: {metric} | alternative={args.alternative}")
    print(f"[INFO] Proposed : {summary['proposed_dir']}")
    print(f"[INFO] Baseline : {summary['baseline_dir']}")
    print(f"[INFO] Proposed source : {proposed_source}")
    print(f"[INFO] Baseline source : {baseline_source}")
    print(f"[INFO] Shared samples : {summary['n_total_shared']}")
    print(f"[INFO] Used samples   : {summary['n_used']} (dropped NaNs: {summary['n_dropped_nan']})")
    if missing_proposed:
        print(f"[WARN] Missing in proposed ({len(missing_proposed)}): {', '.join(missing_proposed[:5])}")
    if missing_baseline:
        print(f"[WARN] Missing in baseline ({len(missing_baseline)}): {', '.join(missing_baseline[:5])}")
    print(
        "[INFO] Proposed mean/median: "
        f"{summary['proposed_mean']:.6f} / {summary['proposed_median']:.6f}"
    )
    print(
        "[INFO] Baseline mean/median: "
        f"{summary['baseline_mean']:.6f} / {summary['baseline_median']:.6f}"
    )
    print(f"[RESULT] Wilcoxon statistic={summary['statistic']:.6f}, p-value={summary['p_value']:.6g}")

    if args.output:
        _write_output(args.output, summary)
        print(f"[INFO] Wrote summary: {args.output}")


if __name__ == "__main__":
    main()
