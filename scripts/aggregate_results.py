#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import autorootcwd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from src.metrics.hd95 import hd95 as compute_hd95_metric

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


matplotlib.use("Agg")


LATEX_TABLE_ARRAYSTRETCH = "1.5"
# NOTE: "decoder_guided" is kept as a legacy alias for old result folders.
KNOWN_CONN_FUSIONS = ("conv_residual", "scaled_sum", "gate", "dg_direct", "dg_dsf", "dsf", "dg", "decoder_guided")
ALLOWED_LABEL_MODES = ("binary", "dist", "dist_inverted")
CONN_FUSION_SORT_ORDER = {
    "none": 0,
    "conv_residual": 1,
    "gate": 2,
    "scaled_sum": 3,
    "dg": 4,
    "dg_direct": 5,
    "dsf": 6,
    "dg_dsf": 7,
    "decoder_guided": 4,
}

LEGACY_CONN_FUSION_ALIASES = {
    "decoder_guided": "dg",
}


def normalize_conn_fusion_name(conn_fusion: object) -> str:
    conn_fusion_name = str(conn_fusion if conn_fusion is not None else "none")
    return LEGACY_CONN_FUSION_ALIASES.get(conn_fusion_name, conn_fusion_name)


def normalize_fusion_branches_name(fusion_branches: object) -> str:
    name = str(fusion_branches if fusion_branches is not None else "none").strip().lower()
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1].replace("'", "").replace('"', "").strip()
    if name in {"i", "io", "im", "imo", "ip", "if", "iop", "iopf", "imp", "impf"}:
        return name
    return "none"


def normalize_decoder_fusion_name(decoder_fusion: object) -> str:
    name = str(decoder_fusion if decoder_fusion is not None else "none").strip()
    return name if name else "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate DconnNet results and export CSV/LaTeX/PDF."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="output",
        help=(
            "Result root directory. Supported layouts: "
            "<input-root>/<experiment>/<fold>/<input-name>, "
            "<input-root>/<experiment>/final_results_<fold>.csv, "
            "<input-root>/<scope>/<experiment>/final_results_<fold>.csv "
            "(for example output/5folds/<experiment>/final_results_<fold>.csv), "
            "<input-root>/<fold>/<input-name>, or <input-root>/final_results_<fold>.csv. "
            "Use a comma-separated list to aggregate multiple roots."
        ),
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="1",
        help="Comma-separated fold ids to aggregate (example: 1,2,3,4,5).",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="final_results_{fold}.csv",
        help=(
            "Input CSV file name pattern in each fold directory/root. "
            "If missing, legacy results_{fold}.csv is also checked automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/summary",
        help="Directory where summary CSV/TEX/PDF will be written.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="summary",
        help="Output file stem (without extension).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Aggregate all counted datasets into combined outputs. "
            "By default, combined outputs include only DRIVE, CHASE, and "
            "OCTA-family datasets."
        ),
    )
    parser.add_argument(
        "--sample-vis-count",
        type=int,
        default=2,
        help=(
            "Number of highest- and lowest-Dice samples to visualize from the "
            "reference model. Set to 0 to disable sample visualization export."
        ),
    )
    parser.add_argument(
        "--max-vis-models",
        type=int,
        default=5,
        help=(
            "Maximum number of models to include in the sample visualization grid. "
            "Comparing too many models (e.g., > 10) significantly increases "
            "the execution time of matplotlib rendering."
        ),
    )
    parser.add_argument(
        "--move-older-to-legacy",
        action="store_true",
        help=(
            "After logical deduplication, move dropped (older) experiment folders to "
            "output/legacy/<dataset>/.  When the 'newer' folder has no parseable CSV "
            "but the 'older' does, the broken newer folder is moved instead."
        ),
    )
    return parser.parse_args()


def parse_input_roots(input_root: str) -> List[str]:
    roots: List[str] = []
    seen: Set[str] = set()
    for raw in str(input_root).split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.normpath(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        roots.append(normalized)
    return roots


def _to_float(value: str) -> float:
    value = value.strip()
    if value.lower() in {"", "nan"}:
        return float("nan")
    return float(value)


def _safe_mean(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _safe_std(values: List[float], mean: float) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) <= 1:
        return 0.0
    var = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    return math.sqrt(var)


_SAMPLE_STD_METRIC_KEYS = (
    "dice", "jac", "cldice",
    "betti_error_0", "betti_error_1", "hd95",
    "precision", "accuracy",
)


def compute_sample_std_from_csv(path: str) -> Dict[str, float]:
    """Compute per-sample std from the sample rows of a final_results.csv or metrics_*.csv.

    Returns a dict keyed as ``best_<metric>`` matching the std_row layout
    returned by aggregate_root().  Returns NaN for metrics with fewer than
    2 valid samples.
    """
    samples: Dict[str, List[float]] = {k: [] for k in _SAMPLE_STD_METRIC_KEYS}
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. Determine if this is a sample row
                res_type = row.get("result_type")
                if res_type:
                    # Standard final_results.csv format
                    if res_type != "sample":
                        continue
                else:
                    # Possibly clDice-style baseline CSV with 'filename' column
                    fname = str(row.get("filename", row.get("Sample", ""))).strip().upper()
                    if not fname or fname in {"MEAN", "STD"}:
                        continue

                # 2. Extract metrics with fallback mappings
                for k in _SAMPLE_STD_METRIC_KEYS:
                    v_str = row.get(k)
                    if v_str is None:
                        # Fallback mappings for clDice baseline CSVs
                        if k == "jac":
                            v_str = row.get("iou") or row.get("Jac")
                        elif k == "betti_error_0":
                            v_str = row.get("betti0_error") or row.get("Betti0")
                        elif k == "betti_error_1":
                            v_str = row.get("betti1_error") or row.get("Betti1")

                    v_str = str(v_str or "nan").strip()
                    v = float("nan") if v_str.lower() in {"", "nan"} else float(v_str)
                    if not math.isnan(v):
                        samples[k].append(v)
    except (OSError, ValueError):
        pass

    result: Dict[str, float] = {}
    for k in _SAMPLE_STD_METRIC_KEYS:
        vals = samples[k]
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            result[f"best_{k}"] = _safe_std(vals, mean)
        else:
            result[f"best_{k}"] = float("nan")
    return result


def _safe_duration_std(values: List[float], mean: float) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return 0.0
    var = sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)
    return math.sqrt(var)


def _row_float_from_header(
    row: List[str],
    header_map: Dict[str, int],
    key: str,
    default: float = float("nan"),
) -> float:
    idx = header_map.get(key)
    if idx is None or idx >= len(row):
        return default
    value = row[idx].strip()
    if value == "":
        return default
    return _to_float(value)


def _dict_row_float(
    row: Dict[str, str],
    key: str,
    default: float = float("nan"),
) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return default
    return _to_float(value)


def parse_hms_duration(value: str) -> float:
    value = value.strip()
    if value == "":
        return float("nan")
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid HH:MM:SS duration: {value}")
    hours, minutes, seconds = [int(part) for part in parts]
    return float(hours * 3600 + minutes * 60 + seconds)


def format_hms_duration(value: float) -> str:
    if math.isnan(float(value)):
        return ""
    total_seconds = max(0, int(round(float(value))))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _fmt_duration_csv(value: object) -> str:
    if isinstance(value, (int, float)):
        return format_hms_duration(float(value))
    text = str(value).strip()
    return text


def _fmt_duration_latex(value: object) -> str:
    text = _fmt_duration_csv(value)
    return text if text else "N/A"


def _normalize_dataset_name(dataset_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", dataset_name.lower())


def _is_counted_dataset_name(dataset_name: object) -> bool:
    return _normalize_dataset_name(str(dataset_name)) != "trash"


def _dataset_summary_group(dataset_name: str) -> Tuple[int, str]:
    normalized = _normalize_dataset_name(dataset_name)
    if normalized == "cremi":
        return 0, "CREMI"
    if normalized == "drive":
        return 1, "DRIVE"
    if normalized in {"chase", "chasedb1"}:
        return 2, "CHASE"
    if normalized == "deepglobe":
        return 3, "DeepGlobe"
    if normalized == "deepcrack":
        return 4, "DeepCrack"
    if normalized.startswith("isic"):
        return 5, "ISIC"
    if "octa" in normalized and any(tag in normalized for tag in ("3m", "6m")):
        return 6, "OCTA3M&6M"
    if normalized == "xacv":
        return 7, "XACV"
    if normalized == "fives":
        return 8, "FIVES"
    return 9, "Other datasets"


def _dataset_group_sort_key(dataset_name: str) -> Tuple[int, str, List[object]]:
    group_order, group_name = _dataset_summary_group(dataset_name)
    return group_order, group_name, natural_sort_key(dataset_name)


def _append_dataset_group_heading(lines: List[str], dataset_name: str) -> None:
    _, group_name = _dataset_summary_group(dataset_name)
    lines.append(rf"\section*{{{escape_latex_text(group_name)}}}")


def _append_page_break(lines: List[str], has_previous_content: bool) -> bool:
    if has_previous_content:
        lines.append(r"\clearpage")
    return True


def is_octa_dataset_name(dataset_name: object) -> bool:
    normalized = _normalize_dataset_name(str(dataset_name))
    return "octa" in normalized and any(tag in normalized for tag in ("3m", "6m"))


def should_include_dataset_in_default_aggregate(dataset_name: object) -> bool:
    normalized = _normalize_dataset_name(str(dataset_name))
    return (
        normalized == "drive"
        or normalized in {"chase", "chasedb1"}
        or normalized == "deepglobe"
        or normalized == "deepcrack"
        or normalized == "xacv"
        or normalized == "fives"
        or is_octa_dataset_name(dataset_name)
    )


def should_include_dataset_in_aggregate(dataset_name: object, include_all: bool) -> bool:
    if not _is_counted_dataset_name(dataset_name):
        return False
    if include_all:
        return True
    return should_include_dataset_in_default_aggregate(dataset_name)


def filter_rows_for_aggregate_scope(
    rows: List[Dict[str, object]],
    include_all: bool,
) -> List[Dict[str, object]]:
    return [
        row for row in rows
        if should_include_dataset_in_aggregate(row.get("dataset", "unknown"), include_all)
    ]


def _format_conn_table_label(conn_num: object, conn_layout: object = "default") -> str:
    conn_text = str(conn_num)
    if str(conn_layout) == "out8" and conn_text == "8":
        return "8'"
    return conn_text


def _conn_layout_sort_key(conn_layout: object) -> Tuple[int, str]:
    layout_text = str(conn_layout)
    order = {
        "default": 0,
        "standard8": 0,
        "full24": 0,
        "out8": 1,
    }.get(layout_text, 9)
    return order, layout_text.lower()


def _load_binary_checkpoint_map(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        image_arr = np.asarray(image.convert("L"), dtype=np.float32)
    if image_arr.size == 0:
        return image_arr
    if float(np.max(image_arr)) > 1.0:
        image_arr = image_arr / 255.0
    return (image_arr > 0.5).astype(np.float32)


@lru_cache(maxsize=None)
def compute_hd95_from_checkpoint_batches(model_dir: str, epoch: Optional[int]) -> float:
    candidate_roots = [model_dir]
    models_dir = os.path.join(model_dir, "models")
    if os.path.isdir(models_dir):
        candidate_roots.insert(0, models_dir)

    checkpoint_root = None
    search_base_dir = None
    for candidate_root in candidate_roots:
        candidate_checkpoint_root = os.path.join(candidate_root, "checkpoint_batches")
        if os.path.isdir(candidate_checkpoint_root):
            checkpoint_root = candidate_checkpoint_root
            search_base_dir = candidate_root
            break
    if checkpoint_root is None:
        return float("nan")

    resolved_epoch: Optional[int] = None
    if epoch is not None and not math.isnan(float(epoch)):
        resolved_epoch, _ = resolve_visualization_epoch(str(search_base_dir), int(epoch))
    if resolved_epoch is None:
        epochs = discover_checkpoint_epochs(str(search_base_dir))
        if not epochs:
            return float("nan")
        resolved_epoch = epochs[-1]

    epoch_dir = os.path.join(checkpoint_root, f"epoch_{int(resolved_epoch):03d}")
    pred_dir = os.path.join(epoch_dir, "pred")
    mask_dir = os.path.join(epoch_dir, "mask")
    if not os.path.isdir(pred_dir) or not os.path.isdir(mask_dir):
        return float("nan")

    pred_files = {
        name for name in os.listdir(pred_dir)
        if name.lower().endswith(".png") and os.path.isfile(os.path.join(pred_dir, name))
    }
    mask_files = {
        name for name in os.listdir(mask_dir)
        if name.lower().endswith(".png") and os.path.isfile(os.path.join(mask_dir, name))
    }
    common_files = sorted(pred_files & mask_files)
    if not common_files:
        return float("nan")

    values: List[float] = []
    for file_name in common_files:
        pred_path = os.path.join(pred_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)
        pred_arr = _load_binary_checkpoint_map(pred_path)
        mask_arr = _load_binary_checkpoint_map(mask_path)
        if pred_arr.shape != mask_arr.shape or pred_arr.size == 0:
            continue
        value = float(compute_hd95_metric(pred_arr, mask_arr))
        if not math.isnan(value):
            values.append(value)

    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def parse_final_summary_csv(path: str) -> Optional[Dict[str, object]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row is None:
            return None

    if "dice" not in row or "jac" not in row or "cldice" not in row:
        return None

    eval_epoch_raw = row.get("eval_epoch", "").strip()
    eval_epoch = int(eval_epoch_raw) if eval_epoch_raw.isdigit() else 0

    elapsed_raw = row.get("elapsed_hms", "").strip()
    elapsed_seconds = (
        parse_hms_duration(elapsed_raw) if elapsed_raw != "" else float("nan")
    )

    best_hd95 = _to_float(str(row.get("hd95", "nan")))
    if math.isnan(best_hd95):
        best_hd95 = compute_hd95_from_checkpoint_batches(os.path.dirname(path), eval_epoch)

    res = {
        "best_epoch": float(eval_epoch),
        "best_dice": _to_float(str(row.get("dice", "nan"))),
        "best_jac": _to_float(str(row.get("jac", "nan"))),
        "best_cldice": _to_float(str(row.get("cldice", "nan"))),
        "best_precision": _dict_row_float(row, "precision"),
        "best_accuracy": _dict_row_float(row, "accuracy"),
        "best_betti_error_0": _to_float(str(row.get("betti_error_0", "nan"))),
        "best_betti_error_1": _to_float(str(row.get("betti_error_1", "nan"))),
        "best_hd95": best_hd95,
        "train_elapsed_seconds": elapsed_seconds,
        "train_elapsed_hms": format_hms_duration(elapsed_seconds),
    }
    # Extract standard deviations if present
    for k in _SAMPLE_STD_METRIC_KEYS:
        std_key = f"{k}_std"
        if std_key in row:
            res[f"best_{k}_std"] = _to_float(str(row[std_key]))
    return res


def parse_fold_csv(path: str) -> Dict[str, object]:
    final_summary = parse_final_summary_csv(path)
    if final_summary is not None:
        return final_summary

    epoch_rows: Dict[int, Dict[str, float]] = {}
    summary_epoch: Optional[int] = None
    summary_dice: Optional[float] = None
    header_map: Optional[Dict[str, int]] = None
    last_elapsed_seconds = float("nan")

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [item.strip() for item in row]
            if not any(item != "" for item in row):
                continue
            if header_map is None and row[0].lower() == "epoch":
                header_map = {
                    item.lower(): idx for idx, item in enumerate(row) if item != ""
                }
                continue

            if row[0] == "":
                continue

            try:
                epoch = int(row[0])
            except ValueError:
                continue

            if (
                header_map is not None and
                "dice" in header_map and
                "jac" in header_map and
                "cldice" in header_map
            ):
                required_idx = max(
                    header_map["dice"],
                    header_map["jac"],
                    header_map["cldice"],
                )
                if len(row) > required_idx:
                    dice = _to_float(row[header_map["dice"]])
                    jac = _to_float(row[header_map["jac"]])
                    cldice = _to_float(row[header_map["cldice"]])
                    precision = _row_float_from_header(row, header_map, "precision")
                    accuracy = _row_float_from_header(row, header_map, "accuracy")
                    betti_error_0 = _row_float_from_header(row, header_map, "betti_error_0")
                    betti_error_1 = _row_float_from_header(row, header_map, "betti_error_1")
                    hd95_val = _row_float_from_header(row, header_map, "hd95")
                    if (
                        "elapsed_hms" in header_map and
                        header_map["elapsed_hms"] < len(row)
                    ):
                        elapsed_seconds = parse_hms_duration(row[header_map["elapsed_hms"]])
                        if not math.isnan(elapsed_seconds):
                            last_elapsed_seconds = elapsed_seconds
                    epoch_rows[epoch] = {
                        "dice": dice,
                        "jac": jac,
                        "cldice": cldice,
                        "precision": precision,
                        "accuracy": accuracy,
                        "betti_error_0": betti_error_0,
                        "betti_error_1": betti_error_1,
                        "hd95": hd95_val,
                    }
                    continue

            if len(row) >= 4:
                dice = _to_float(row[1])
                jac = _to_float(row[2])
                cldice = _to_float(row[3])
                precision = float("nan")
                accuracy = float("nan")
                betti_error_0 = _to_float(row[6]) if len(row) >= 7 and row[6] != "" else float("nan")
                betti_error_1 = _to_float(row[7]) if len(row) >= 8 and row[7] != "" else float("nan")
                epoch_rows[epoch] = {
                    "dice": dice,
                    "jac": jac,
                    "cldice": cldice,
                    "precision": precision,
                    "accuracy": accuracy,
                    "betti_error_0": betti_error_0,
                    "betti_error_1": betti_error_1,
                    "hd95": float("nan"),
                }
            elif len(row) >= 2:
                summary_epoch = int(row[0])
                summary_dice = _to_float(row[1])

    if not epoch_rows:
        return None

    if summary_epoch is None or summary_dice is None:
        best_epoch, best_metrics = max(
            epoch_rows.items(), key=lambda kv: kv[1]["dice"]
        )
        best_dice = best_metrics["dice"]
        best_jac = best_metrics["jac"]
        best_cldice = best_metrics["cldice"]
        best_precision = best_metrics.get("precision", float("nan"))
        best_accuracy = best_metrics.get("accuracy", float("nan"))
        best_betti_error_0 = best_metrics["betti_error_0"]
        best_betti_error_1 = best_metrics["betti_error_1"]
        best_hd95 = best_metrics.get("hd95", float("nan"))
    else:
        best_epoch = summary_epoch
        best_dice = summary_dice
        if best_epoch in epoch_rows:
            best_jac = epoch_rows[best_epoch]["jac"]
            best_cldice = epoch_rows[best_epoch]["cldice"]
            best_precision = epoch_rows[best_epoch].get("precision", float("nan"))
            best_accuracy = epoch_rows[best_epoch].get("accuracy", float("nan"))
            best_betti_error_0 = epoch_rows[best_epoch]["betti_error_0"]
            best_betti_error_1 = epoch_rows[best_epoch]["betti_error_1"]
            best_hd95 = epoch_rows[best_epoch].get("hd95", float("nan"))
        else:
            best_jac, best_cldice = float("nan"), float("nan")
            best_precision, best_accuracy = float("nan"), float("nan")
            best_betti_error_0, best_betti_error_1 = float("nan"), float("nan")
            best_hd95 = float("nan")

        if math.isnan(best_hd95):
            best_hd95 = compute_hd95_from_checkpoint_batches(os.path.dirname(path), best_epoch)

    return {
        "best_epoch": float(best_epoch),
        "best_dice": best_dice,
        "best_jac": best_jac,
        "best_cldice": best_cldice,
        "best_precision": best_precision,
        "best_accuracy": best_accuracy,
        "best_betti_error_0": best_betti_error_0,
        "best_betti_error_1": best_betti_error_1,
        "best_hd95": best_hd95,
        "train_elapsed_seconds": last_elapsed_seconds,
        "train_elapsed_hms": format_hms_duration(last_elapsed_seconds),
    }


def build_fold_input_name_candidates(
    input_name: str,
    fold: str,
    allow_legacy_results: bool = True,
) -> List[str]:
    candidates: List[str] = []
    if "{fold}" in input_name:
        candidates.append(input_name.format(fold=fold))
    else:
        candidates.append(input_name)

    # Keep backward compatibility with legacy per-epoch result summaries.
    candidates.append(f"final_results_{fold}.csv")
    if allow_legacy_results:
        candidates.append(f"results_{fold}.csv")
    if str(fold) == "1":
        candidates.append("final_results.csv")
        if allow_legacy_results:
            candidates.append("results.csv")

    unique_candidates: List[str] = []
    seen = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        unique_candidates.append(name)
    return unique_candidates


def resolve_fold_csv_path(
    input_root: str,
    fold: str,
    input_name: str,
    allow_legacy_results: bool = True,
) -> str:
    checked: List[str] = []
    candidate_names = build_fold_input_name_candidates(
        input_name,
        fold,
        allow_legacy_results=allow_legacy_results,
    )

    for candidate_name in candidate_names:
        fold_dir_candidate = os.path.join(input_root, fold, candidate_name)
        checked.append(fold_dir_candidate)
        if os.path.isfile(fold_dir_candidate):
            return fold_dir_candidate

    for candidate_name in candidate_names:
        root_candidate = os.path.join(input_root, candidate_name)
        checked.append(root_candidate)
        if os.path.isfile(root_candidate):
            return root_candidate

    raise FileNotFoundError(
        "Missing fold CSV for fold "
        f"{fold}. Checked: {', '.join(checked)}"
    )


def discover_available_folds(
    input_root: str,
    input_name: str,
    allow_legacy_results: bool = True,
) -> List[str]:
    discovered = set()
    if not os.path.isdir(input_root):
        return []

    single_run_candidates = ["final_results.csv"]
    if allow_legacy_results:
        single_run_candidates.append("results.csv")
    if any(os.path.isfile(os.path.join(input_root, name)) for name in single_run_candidates):
        discovered.add("1")

    for entry in os.listdir(input_root):
        entry_path = os.path.join(input_root, entry)
        if not entry.isdigit() or not os.path.isdir(entry_path):
            continue

        candidate_names = build_fold_input_name_candidates(
            input_name,
            entry,
            allow_legacy_results=allow_legacy_results,
        )
        if any(os.path.isfile(os.path.join(entry_path, name)) for name in candidate_names):
            discovered.add(entry)

    if "{fold}" in input_name:
        prefix, suffix = input_name.split("{fold}")
        for entry in os.listdir(input_root):
            if not (entry.startswith(prefix) and entry.endswith(suffix)):
                continue
            fold = entry[len(prefix): len(entry) - len(suffix) if len(suffix) > 0 else len(entry)]
            if fold.isdigit():
                discovered.add(fold)

    for entry in os.listdir(input_root):
        if not entry.startswith("final_results_") or not entry.endswith(".csv"):
            if allow_legacy_results:
                match = re.fullmatch(r"results_(\d+)\.csv", entry)
                if match is None:
                    continue
                fold = match.group(1)
            else:
                continue
        else:
            fold = entry[len("final_results_"):-len(".csv")]
        if fold.isdigit():
            discovered.add(fold)

    return sorted(discovered, key=int)


def parse_scope_fold_count(name: str) -> Optional[int]:
    match = re.fullmatch(r"(\d+)folds?", name.lower())
    if match is None:
        return None
    return int(match.group(1))


def extract_scope_info(path: str) -> Tuple[str, object]:
    for part in reversed(os.path.normpath(path).split(os.sep)):
        fold_count = parse_scope_fold_count(part)
        if fold_count is not None:
            return part.lower(), fold_count
    return "direct", "NA"


def discover_nested_target_root_infos(
    input_root: str,
    input_name: str,
    max_depth: int = 3,
) -> List[Dict[str, object]]:
    target_infos: List[Dict[str, object]] = []
    pending: List[Tuple[str, int]] = [(input_root, 0)]
    seen_dirs: Set[str] = set()

    while pending:
        current_dir, depth = pending.pop(0)
        normalized_dir = os.path.normpath(current_dir)
        if normalized_dir in seen_dirs:
            continue
        seen_dirs.add(normalized_dir)

        rel_path = os.path.relpath(current_dir, input_root)
        rel_parts = [] if rel_path in {".", ""} else rel_path.split(os.sep)
        if "_smoke" in rel_parts or "legacy" in rel_parts:
            continue

        if depth > 0:
            available_folds = discover_available_folds(current_dir, input_name)
            if available_folds:
                scope_fold_count = None
                for part in rel_parts:
                    scope_fold_count = parse_scope_fold_count(part)
                    if scope_fold_count is not None:
                        break
                target_infos.append(
                    {
                        "root": current_dir,
                        "available_folds": available_folds,
                        "scope_fold_count": scope_fold_count,
                    }
                )
                continue

        if depth >= max_depth:
            continue

        try:
            entries = sorted(os.listdir(current_dir))
        except OSError:
            continue

        for entry in entries:
            entry_path = os.path.join(current_dir, entry)
            if os.path.isdir(entry_path):
                pending.append((entry_path, depth + 1))

    return target_infos


def select_target_root_infos(
    input_root: str,
    target_infos: List[Dict[str, object]],
    requested_folds: List[str],
) -> List[Dict[str, object]]:
    if not target_infos:
        return []

    scope_names = sorted(
        {
            f"{int(info['scope_fold_count'])}-fold"
            for info in target_infos
            if isinstance(info.get("scope_fold_count"), int)
        },
        key=natural_sort_key,
    )
    if len(scope_names) > 1:
        print(
            f"[INFO] {input_root}: multiple output scopes detected; "
            f"including {', '.join(scope_names)} experiment roots for requested folds "
            f"{', '.join(requested_folds)}."
        )

    return sorted(
        target_infos,
        key=lambda item: (
            int(item["scope_fold_count"]) if isinstance(item.get("scope_fold_count"), int) else 10 ** 9,
            str(item["root"]),
        ),
    )


def discover_target_roots(input_root: str, folds: List[str], input_name: str) -> List[str]:
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root directory not found: {input_root}")

    available_in_root = discover_available_folds(input_root, input_name)
    target_infos = discover_nested_target_root_infos(input_root, input_name)
    if target_infos:
        nested_roots = [
            str(info["root"])
            for info in select_target_root_infos(input_root, target_infos, folds)
        ]
        if available_in_root:
            non_fold_nested_roots: List[str] = []
            for root in nested_roots:
                rel_parts = os.path.relpath(root, input_root).split(os.sep)
                if (
                    len(rel_parts) == 1
                    and rel_parts[0].isdigit()
                    and not os.path.isfile(os.path.join(root, "config.yaml"))
                ):
                    continue
                non_fold_nested_roots.append(root)
            if non_fold_nested_roots:
                print(
                    f"[INFO] {input_root}: root-level result CSVs detected, but nested experiment "
                    "roots are also present; aggregating nested experiment folders instead."
                )
                return non_fold_nested_roots

        if available_in_root:
            return [input_root]
        return [
            str(info["root"])
            for info in select_target_root_infos(input_root, target_infos, folds)
        ]

    if available_in_root:
        return [input_root]

    raise FileNotFoundError(
        f"No fold CSV files were found in '{input_root}' or its nested subdirectories."
    )


def root_has_completed_final_results(input_root: str, folds: List[str], input_name: str) -> bool:
    for fold in folds:
        try:
            resolve_fold_csv_path(
                input_root,
                fold,
                input_name,
                allow_legacy_results=True,
            )
        except FileNotFoundError:
            return False
    return True


def resolve_folds_for_root(input_root: str, requested_folds: List[str], input_name: str) -> List[str]:
    missing = []
    for fold in requested_folds:
        try:
            resolve_fold_csv_path(input_root, fold, input_name, allow_legacy_results=True)
        except FileNotFoundError:
            missing.append(fold)

    if not missing:
        return requested_folds

    auto_folds = discover_available_folds(input_root, input_name, allow_legacy_results=False)
    if not auto_folds:
        raise FileNotFoundError(
            f"No completed folds in '{input_root}'. Missing requested folds: {', '.join(missing)}"
        )
    scope_name, scope_fold_count = extract_scope_info(input_root)
    log_tag = "WARN"
    if isinstance(scope_fold_count, int) and scope_fold_count == len(auto_folds):
        log_tag = "INFO"

    print(
        f"[{log_tag}] {input_root}: missing requested folds ({', '.join(missing)}). "
        f"Using detected folds: {', '.join(auto_folds)}"
        + (f" within {scope_name} scope." if log_tag == "INFO" else "")
    )
    return auto_folds


def aggregate_root(input_root: str, folds: List[str], input_name: str) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    fold_rows: List[Dict[str, object]] = []
    for fold in folds:
        csv_path = resolve_fold_csv_path(input_root, fold, input_name, allow_legacy_results=True)
        metrics = parse_fold_csv(csv_path)
        if metrics is None:
            print(f"[WARN] {input_root} fold {fold}: failed to parse metrics from CSV: {csv_path}")
            continue
        metrics["fold"] = float(fold)
        fold_rows.append(metrics)

    epochs = [row["best_epoch"] for row in fold_rows]
    dices = [row["best_dice"] for row in fold_rows]
    jacs = [row["best_jac"] for row in fold_rows]
    cldices = [row["best_cldice"] for row in fold_rows]
    precisions = [row["best_precision"] for row in fold_rows]
    accuracies = [row["best_accuracy"] for row in fold_rows]
    betti_error_0s = [row["best_betti_error_0"] for row in fold_rows]
    betti_error_1s = [row["best_betti_error_1"] for row in fold_rows]
    hd95s = [row.get("best_hd95", float("nan")) for row in fold_rows]
    elapsed_seconds = [float(row["train_elapsed_seconds"]) for row in fold_rows]

    elapsed_mean_seconds = _safe_mean(elapsed_seconds)
    elapsed_std_seconds = _safe_duration_std(elapsed_seconds, elapsed_mean_seconds)

    mean_row = {
        "best_epoch": _safe_mean(epochs),
        "best_dice": _safe_mean(dices),
        "best_jac": _safe_mean(jacs),
        "best_cldice": _safe_mean(cldices),
        "best_precision": _safe_mean(precisions),
        "best_accuracy": _safe_mean(accuracies),
        "best_betti_error_0": _safe_mean(betti_error_0s),
        "best_betti_error_1": _safe_mean(betti_error_1s),
        "best_hd95": _safe_mean(hd95s),
        "train_elapsed_seconds": elapsed_mean_seconds,
        "train_elapsed_hms": format_hms_duration(elapsed_mean_seconds),
    }
    std_row = {
        "best_epoch": _safe_std(epochs, mean_row["best_epoch"]),
        "best_dice": _safe_std(dices, mean_row["best_dice"]),
        "best_jac": _safe_std(jacs, mean_row["best_jac"]),
        "best_cldice": _safe_std(cldices, mean_row["best_cldice"]),
        "best_precision": _safe_std(precisions, mean_row["best_precision"]),
        "best_accuracy": _safe_std(accuracies, mean_row["best_accuracy"]),
        "best_betti_error_0": _safe_std(betti_error_0s, mean_row["best_betti_error_0"]),
        "best_betti_error_1": _safe_std(betti_error_1s, mean_row["best_betti_error_1"]),
        "best_hd95": _safe_std(hd95s, mean_row["best_hd95"]),
        "train_elapsed_seconds": elapsed_std_seconds,
        "train_elapsed_hms": format_hms_duration(elapsed_std_seconds),
    }

    # If only one fold is present, fallback to the internal sample-wise STD of that fold if available.
    if len(fold_rows) == 1:
        row = fold_rows[0]
        for k in _SAMPLE_STD_METRIC_KEYS:
            std_val = row.get(f"best_{k}_std")
            if std_val is not None and not math.isnan(std_val):
                std_row[f"best_{k}"] = std_val

    return fold_rows, mean_row, std_row


def parse_sample_metrics_csv(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "batch": int(row["batch"]),
                    "sample_in_batch": int(row["sample_in_batch"]),
                    "sample_name": row["sample_name"],
                    "val_loss": _to_float(row["val_loss"]),
                    "dice": _to_float(row["dice"]),
                    "jac": _to_float(row["jac"]),
                    "cldice": _to_float(row["cldice"]),
                }
            )
    return rows


def discover_checkpoint_epochs(model_dir: str) -> List[int]:
    checkpoint_root = os.path.join(model_dir, "checkpoint_batches")
    if not os.path.isdir(checkpoint_root):
        return []

    epochs: List[int] = []
    for entry in os.listdir(checkpoint_root):
        match = re.fullmatch(r"epoch_(\d+)", entry)
        if match is not None:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def resolve_visualization_epoch(model_dir: str, target_epoch: int) -> Tuple[Optional[int], bool]:
    epochs = discover_checkpoint_epochs(model_dir)
    if not epochs:
        return None, False
    if target_epoch in epochs:
        return target_epoch, True

    nearest = min(epochs, key=lambda epoch: (abs(epoch - target_epoch), -epoch))
    return nearest, False


def find_fold_row(fold_rows: List[Dict[str, object]], fold: int) -> Optional[Dict[str, object]]:
    for row in fold_rows:
        if int(row["fold"]) == int(fold):
            return row
    return None


def build_model_candidate(summary: Dict[str, object], fold_row: Dict[str, object]) -> Dict[str, object]:
    fold = int(fold_row["fold"])
    model_dir = os.path.join(str(summary["root"]), "models", str(fold))
    if not os.path.isdir(model_dir):
        model_dir = os.path.join(str(summary["root"]), "models")
    sample_csv = os.path.join(model_dir, "test_sample_metrics.csv")
    sample_rows = parse_sample_metrics_csv(sample_csv) if os.path.isfile(sample_csv) else []
    best_epoch = int(fold_row["best_epoch"])
    vis_epoch, is_exact = resolve_visualization_epoch(model_dir, best_epoch)
    exp_meta = parse_experiment_metadata(os.path.basename(os.path.normpath(str(summary["root"]))))
    _, fold_scope_count = extract_scope_info(str(summary["root"]))
    config_folds = fold_scope_count if isinstance(fold_scope_count, int) else len(summary["folds"])

    return {
        "root": summary["root"],
        "root_name": summary["root_name"],
        "experiment": exp_meta["experiment"],
        "conn_num": exp_meta["conn_num"],
        "conn_layout": exp_meta["conn_layout"],
        "loss": exp_meta["loss"],
        "config_folds": config_folds,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_dice": float(fold_row["best_dice"]),
        "model_dir": model_dir,
        "sample_csv": sample_csv,
        "sample_rows": sample_rows,
        "vis_epoch": vis_epoch,
        "vis_epoch_exact": is_exact,
    }


def choose_visualization_models(
    root_summaries: List[Dict[str, object]],
) -> Tuple[Optional[Dict[str, object]], List[Dict[str, object]]]:
    if not root_summaries:
        return None, []

    ranked_summaries = sorted(
        root_summaries,
        key=lambda item: float(item["mean_row"]["best_dice"]),
        reverse=True,
    )
    reference_summary = ranked_summaries[0]
    reference_fold_row = max(
        reference_summary["fold_rows"],
        key=lambda row: float(row["best_dice"]),
    )
    reference_model = build_model_candidate(reference_summary, reference_fold_row)
    reference_fold = int(reference_fold_row["fold"])

    models: List[Dict[str, object]] = [reference_model]
    for summary in ranked_summaries[1:]:
        fold_row = find_fold_row(summary["fold_rows"], reference_fold)
        if fold_row is None:
            fold_row = max(
                summary["fold_rows"],
                key=lambda row: float(row["best_dice"]),
            )
            print(
                f"[WARN] {summary['root_name']}: fold {reference_fold} "
                "not found for sample comparison; using its best fold instead."
            )
        models.append(build_model_candidate(summary, fold_row))

    return reference_model, models


def filter_sample_rows_for_epoch(sample_rows: List[Dict[str, object]], epoch: int) -> List[Dict[str, object]]:
    return [row for row in sample_rows if int(row["epoch"]) == int(epoch)]


def select_ranked_samples(sample_rows: List[Dict[str, object]], count: int) -> List[Dict[str, object]]:
    if count <= 0:
        return []

    selected: List[Dict[str, object]] = []
    used_names: Set[str] = set()

    top_rows = sorted(sample_rows, key=lambda row: (-float(row["dice"]), str(row["sample_name"])))
    for row in top_rows:
        sample_name = str(row["sample_name"])
        if sample_name in used_names:
            continue
        picked = dict(row)
        picked["sample_group"] = "top"
        selected.append(picked)
        used_names.add(sample_name)
        if len([item for item in selected if item["sample_group"] == "top"]) >= count:
            break

    bottom_rows = sorted(sample_rows, key=lambda row: (float(row["dice"]), str(row["sample_name"])))
    bottom_selected = 0
    for row in bottom_rows:
        sample_name = str(row["sample_name"])
        if sample_name in used_names:
            continue
        picked = dict(row)
        picked["sample_group"] = "bottom"
        selected.append(picked)
        used_names.add(sample_name)
        bottom_selected += 1
        if bottom_selected >= count:
            break

    for idx, row in enumerate(selected, start=1):
        row["sample_rank"] = idx

    return selected


def find_sample_row(sample_rows: List[Dict[str, object]], epoch: int, sample_name: str) -> Optional[Dict[str, object]]:
    for row in sample_rows:
        if int(row["epoch"]) == int(epoch) and str(row["sample_name"]) == sample_name:
            return row
    return None


def checkpoint_image_path(model_dir: str, epoch: Optional[int], kind: str, batch_idx: int) -> Optional[str]:
    if epoch is None:
        return None
    path = os.path.join(
        model_dir,
        "checkpoint_batches",
        f"epoch_{int(epoch):03d}",
        kind,
        f"batch_{int(batch_idx):04d}.png",
    )
    return path if os.path.isfile(path) else None


def load_panel_image(image_path: str, panel_kind: str) -> np.ndarray:
    with Image.open(image_path) as image:
        if panel_kind == "image":
            # Keep input images in RGB and auto-correct likely BGR-encoded snapshots.
            image = image.convert("RGB")
            image_arr = np.asarray(image)
            if image_arr.ndim == 3 and image_arr.shape[-1] == 3:
                red_mean = float(np.mean(image_arr[:, :, 0]))
                blue_mean = float(np.mean(image_arr[:, :, 2]))
                # Fundus-like images should generally not be blue-dominant.
                # If blue is clearly larger than red, treat it as BGR-encoded and swap.
                if blue_mean > red_mean * 1.10:
                    image_arr = image_arr[:, :, ::-1]
            return image_arr

        # Segmentation targets/predictions are binary-like maps; render as single-channel grayscale.
        image = image.convert("L")
        return np.asarray(image)


def render_panel(
    ax,
    image_path: Optional[str],
    title: str,
    panel_kind: str,
    row_label: Optional[str] = None,
) -> None:
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if row_label:
        ax.set_ylabel(row_label, rotation=0, labelpad=52, va="center")

    if image_path is None:
        ax.set_facecolor("#f4f4f4")
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        return

    image = load_panel_image(image_path, panel_kind)
    if panel_kind != "image":
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(image)


def format_model_config_summary(model: Dict[str, object], multiline: bool = False) -> str:
    separator = "\n" if multiline else " | "
    seg_aux_label = _format_segaux_column_value(
        model.get("seg_aux_weight"),
        model.get("seg_aux_variant", "none"),
    )
    fields = [
        f"Exp={model['experiment']}",
        f"Conn={_format_conn_table_label(model['conn_num'], model.get('conn_layout', 'default'))}",
        f"Fusion={_format_fusion_table_label(model.get('conn_fusion', 'none'), model.get('fusion_residual_scale'))}",
        f"SegAux={seg_aux_label or 'none'}",
        f"Loss={model['loss']}",
    ]
    fields.append(f"Folds={model['config_folds']}")
    return separator.join(fields)


def build_sample_visualization_fieldnames(model_count: int) -> List[str]:
    fieldnames = [
        "sample_rank",
        "sample_group",
        "sample_name",
        "reference_dice",
        "reference_jac",
        "reference_cldice",
        "image_path",
        "gt_path",
    ]
    for idx in range(1, model_count + 1):
        fieldnames.extend(
            [
                f"model{idx}_name",
                f"model{idx}_config",
                f"model{idx}_experiment",
                f"model{idx}_conn_num",
                f"model{idx}_loss",
                f"model{idx}_folds",
                f"model{idx}_fold",
                f"model{idx}_best_epoch",
                f"model{idx}_vis_epoch",
                f"model{idx}_vis_epoch_exact",
                f"model{idx}_dice",
                f"model{idx}_pred_path",
            ]
        )
    return fieldnames


def write_sample_visualization_csv(path: str, rows: List[Dict[str, object]], model_count: int) -> None:
    fieldnames = build_sample_visualization_fieldnames(model_count)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_sample_visualization(
    output_dir: str,
    image_output_dir: str,
    output_stem: str,
    root_summaries: List[Dict[str, object]],
    sample_vis_count: int,
    max_vis_models: int = 5,
) -> None:
    if sample_vis_count <= 0:
        return

    reference_model, models = choose_visualization_models(root_summaries)
    if reference_model is None or not models:
        return

    if max_vis_models > 0 and len(models) > max_vis_models:
        models = models[:max_vis_models]

    if reference_model["vis_epoch"] is None:
        print(f"[WARN] {reference_model['model_dir']}: no checkpoint_batches found; skipping sample visualization.")
        return
    if not reference_model["sample_rows"]:
        print(f"[WARN] {reference_model['sample_csv']}: missing sample metrics; skipping sample visualization.")
        return

    reference_rows = filter_sample_rows_for_epoch(reference_model["sample_rows"], int(reference_model["best_epoch"]))
    if not reference_rows:
        print(
            f"[WARN] {reference_model['sample_csv']}: no rows for best epoch {reference_model['best_epoch']}; "
            "skipping sample visualization."
        )
        return

    selected_rows = select_ranked_samples(reference_rows, sample_vis_count)
    if not selected_rows:
        print("[WARN] No sample rows available for visualization.")
        return

    if not bool(reference_model["vis_epoch_exact"]):
        print(
            f"[WARN] {reference_model['root_name']} fold {reference_model['fold']}: "
            f"best epoch {reference_model['best_epoch']} has no saved PNG; "
            f"using nearest checkpoint epoch {reference_model['vis_epoch']}."
        )
    for model in models[1:]:
        if model["vis_epoch"] is None:
            print(f"[WARN] {model['model_dir']}: no checkpoint_batches found; prediction column will be N/A.")
            continue
        if not bool(model["vis_epoch_exact"]):
            print(
                f"[WARN] {model['root_name']} fold {model['fold']}: "
                f"best epoch {model['best_epoch']} has no saved PNG; using nearest checkpoint epoch {model['vis_epoch']}."
            )
        if not model["sample_rows"]:
            print(f"[WARN] {model['sample_csv']}: missing sample metrics; prediction column may be N/A.")

    render_rows: List[Dict[str, object]] = []
    for row in selected_rows:
        sample_name = str(row["sample_name"])
        batch_idx = int(row["batch"])
        image_path = checkpoint_image_path(str(reference_model["model_dir"]), reference_model["vis_epoch"], "image", batch_idx)
        gt_path = checkpoint_image_path(str(reference_model["model_dir"]), reference_model["vis_epoch"], "mask", batch_idx)

        render_row: Dict[str, object] = {
            "sample_rank": int(row["sample_rank"]),
            "sample_group": str(row["sample_group"]),
            "sample_name": sample_name,
            "reference_dice": float(row["dice"]),
            "reference_jac": float(row["jac"]),
            "reference_cldice": float(row["cldice"]),
            "image_path": image_path or "",
            "gt_path": gt_path or "",
        }

        for idx, model in enumerate(models, start=1):
            model_row = None
            model_pred_path = None
            model_dice = float("nan")
            model_vis_epoch = ""
            if model["vis_epoch"] is not None:
                model_vis_epoch = int(model["vis_epoch"])

            if model["vis_epoch"] is not None and model["sample_rows"]:
                model_row = find_sample_row(model["sample_rows"], int(model["best_epoch"]), sample_name)
                if model_row is not None:
                    model_dice = float(model_row["dice"])
                    model_pred_path = checkpoint_image_path(
                        str(model["model_dir"]),
                        model["vis_epoch"],
                        "pred",
                        int(model_row["batch"]),
                    )

            render_row[f"model{idx}_name"] = str(model["root_name"])
            render_row[f"model{idx}_config"] = format_model_config_summary(model)
            render_row[f"model{idx}_experiment"] = str(model["experiment"])
            render_row[f"model{idx}_conn_num"] = _format_conn_table_label(
                model["conn_num"],
                model.get("conn_layout", "default"),
            )
            render_row[f"model{idx}_loss"] = str(model["loss"])
            render_row[f"model{idx}_folds"] = model["config_folds"]
            render_row[f"model{idx}_fold"] = int(model["fold"])
            render_row[f"model{idx}_best_epoch"] = int(model["best_epoch"])
            render_row[f"model{idx}_vis_epoch"] = model_vis_epoch
            render_row[f"model{idx}_vis_epoch_exact"] = bool(model["vis_epoch_exact"])
            render_row[f"model{idx}_dice"] = model_dice
            render_row[f"model{idx}_pred_path"] = model_pred_path or ""

        render_rows.append(render_row)

    vis_csv_out = os.path.join(output_dir, f"{output_stem}_sample_visualization.csv")
    vis_png_out = os.path.join(image_output_dir, f"{output_stem}_sample_visualization.png")
    write_sample_visualization_csv(vis_csv_out, render_rows, len(models))

    fig, axes = plt.subplots(
        nrows=len(render_rows),
        ncols=2 + len(models),
        figsize=(max(14, (2 + len(models)) * 3.7), max(4, len(render_rows) * 4)),
        squeeze=False,
    )

    model_titles = [format_model_config_summary(model, multiline=True) for model in models]
    figure_title = (
        f"Reference: {format_model_config_summary(reference_model)} / fold {reference_model['fold']} "
        f"/ best epoch {reference_model['best_epoch']} (viz epoch {reference_model['vis_epoch']})"
    )
    if len(models) > 1:
        figure_title += f"\nCompared Models: {len(models)}"

    for row_idx, row in enumerate(render_rows):
        row_label = (
            f"Sample {row['sample_rank']}\n"
            f"{row['sample_name']}\n"
            f"{row['sample_group']} dice={row['reference_dice']:.4f}"
        )
        render_panel(
            axes[row_idx][0],
            row["image_path"] or None,
            "Image",
            panel_kind="image",
            row_label=row_label,
        )
        render_panel(axes[row_idx][1], row["gt_path"] or None, "GT", panel_kind="mask")
        for model_idx, title in enumerate(model_titles, start=1):
            render_panel(
                axes[row_idx][model_idx + 1],
                row.get(f"model{model_idx}_pred_path") or None,
                title,
                panel_kind="pred",
            )

    fig.suptitle(figure_title, fontsize=12)
    plt.tight_layout(rect=(0.03, 0.03, 1.0, 0.93))
    fig.savefig(vis_png_out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAMPLE_VIS] CSV: {vis_csv_out}")
    print(f"[SAMPLE_VIS] PNG: {vis_png_out}")


def write_summary_csv(path: str, fold_rows: List[Dict[str, object]], mean_row: Dict[str, object], std_row: Dict[str, object]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fold",
                "best_epoch",
                "best_dice",
                "best_jac",
                "best_cldice",
                "best_betti_error_0",
                "best_betti_error_1",
                "best_hd95",
                "train_elapsed_hms",
            ]
        )
        for row in fold_rows:
            writer.writerow(
                [
                    int(row["fold"]),
                    int(row["best_epoch"]),
                    f'{row["best_dice"]:.4f}',
                    f'{row["best_jac"]:.4f}' if not math.isnan(
                        row["best_jac"]) else "nan",
                    f'{row["best_cldice"]:.4f}' if not math.isnan(
                        row["best_cldice"]) else "nan",
                    f'{row["best_betti_error_0"]:.4f}' if not math.isnan(
                        row["best_betti_error_0"]) else "nan",
                    f'{row["best_betti_error_1"]:.4f}' if not math.isnan(
                        row["best_betti_error_1"]) else "nan",
                    _fmt_float(float(row.get("best_hd95", float("nan")))),
                    _fmt_duration_csv(row.get("train_elapsed_hms", "")),
                ]
            )
        writer.writerow(
            [
                "mean",
                f'{mean_row["best_epoch"]:.4f}',
                f'{mean_row["best_dice"]:.4f}',
                f'{mean_row["best_jac"]:.4f}' if not math.isnan(
                    mean_row["best_jac"]) else "nan",
                f'{mean_row["best_cldice"]:.4f}' if not math.isnan(
                    mean_row["best_cldice"]) else "nan",
                f'{mean_row["best_betti_error_0"]:.4f}' if not math.isnan(
                    mean_row["best_betti_error_0"]) else "nan",
                f'{mean_row["best_betti_error_1"]:.4f}' if not math.isnan(
                    mean_row["best_betti_error_1"]) else "nan",
                _fmt_float(float(mean_row.get("best_hd95", float("nan")))),
                _fmt_duration_csv(mean_row.get("train_elapsed_hms", "")),
            ]
        )
        writer.writerow(
            [
                "std",
                f'{std_row["best_epoch"]:.4f}',
                f'{std_row["best_dice"]:.4f}',
                f'{std_row["best_jac"]:.4f}' if not math.isnan(
                    std_row["best_jac"]) else "nan",
                f'{std_row["best_cldice"]:.4f}' if not math.isnan(
                    std_row["best_cldice"]) else "nan",
                f'{std_row["best_betti_error_0"]:.4f}' if not math.isnan(
                    std_row["best_betti_error_0"]) else "nan",
                f'{std_row["best_betti_error_1"]:.4f}' if not math.isnan(
                    std_row["best_betti_error_1"]) else "nan",
                _fmt_float(float(std_row.get("best_hd95", float("nan")))),
                _fmt_duration_csv(std_row.get("train_elapsed_hms", "")),
            ]
        )


def _fmt_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _fmt_aux_weight(value: float) -> str:
    """Format a weight like 0.3→'0.3', 1.0→'1.0' (always keeps one decimal place)."""
    s = f"{value:g}"
    return s if "." in s else f"{s}.0"


def infer_loss_tag_from_legacy_name(experiment: str) -> str:
    name = experiment.lower()
    if "cl_dice" in name:
        return "cl_dice"
    if "_gjml_sf_l1" in name:
        return "gjml_sf_l1"
    if "_gjml_sj_l1" in name:
        return "gjml_sj_l1"
    if "_smooth_l1" in name:
        return "smooth_l1"
    if "dist" in name:
        return "smooth_l1"
    if "binary" in name:
        return "bce"
    return "unknown"


def _rank_row_indices(rows: List[Dict[str, object]], key: str, higher_is_better: bool = True) -> Tuple[Set[int], Set[int]]:
    indexed_values: List[Tuple[int, float]] = []
    for idx, row in enumerate(rows):
        value = row.get(key)
        if value is None:
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
        indexed_values.append((idx, value_f))

    if not indexed_values:
        return set(), set()

    ranked_values = sorted({value for _, value in indexed_values}, reverse=higher_is_better)
    best_value = ranked_values[0]
    second_value = ranked_values[1] if len(ranked_values) > 1 else None

    best_indices = {idx for idx, value in indexed_values if value == best_value}
    second_indices = (
        {idx for idx, value in indexed_values if second_value is not None and value == second_value}
        if second_value is not None
        else set()
    )
    return best_indices, second_indices


def _fmt_latex_ranked_value(value: float, is_best: bool, is_second: bool) -> str:
    if math.isnan(value):
        return "-"
    text = _fmt_float(value)
    if is_best:
        return rf"\textbf{{{text}}}"
    if is_second:
        return rf"\underline{{{text}}}"
    return text


def natural_sort_key(text: str) -> Tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text))


def segaux_sort_key(value: object) -> Tuple[object, ...]:
    text = str(value).strip()
    if text == "":
        return (2, 1, "")
    if text == "none":
        return (0, 0, 0.0)
    if text.startswith("w"):
        # Parse format like "w0.5" or "w0.5(im)"
        match = re.match(r"w([\d.]+)(?:\(([^)]*)\))?", text)
        if match:
            try:
                numeric_part = float(match.group(1))
                suffix_part = match.group(2) or ""
                return (1, 0, numeric_part, suffix_part)
            except ValueError:
                pass
    try:
        return (1, 0, float(text))
    except ValueError:
        return (1, 1, natural_sort_key(text))


def _fmt_latex_ranked_mean_std_marginal(mean: float, std: float, is_best: bool, is_second: bool) -> Tuple[str, str]:
    if math.isnan(mean):
        return "--", "-"
    mean_txt = f"{mean:.4f}"
    if is_best:
        mean_txt = rf"\textbf{{{mean_txt}}}"
    elif is_second:
        mean_txt = rf"\underline{{{mean_txt}}}"
    std_txt = rf"(\textcolor{{gray}}{{{std:.4f}}})" if not math.isnan(std) else r"\textcolor{gray}{(-)}"
    return mean_txt, std_txt


def sanitize_scope_name_for_filename(scope_name: str) -> str:
    sanitized = scope_name.strip().replace(os.sep, "_")
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", sanitized).strip("_")
    return sanitized if sanitized else "unknown_scope"


def canonical_dataset_name(text: str) -> Optional[str]:
    name = text.lower()
    if name in {"chase", "chasedb1"}:
        return "chase"
    if name == "drive":
        return "drive"
    if name.startswith("isic"):
        return "isic"
    if name == "cremi":
        return "cremi"
    if name == "deepcrack":
        return "deepcrack"
    if name == "xacv":
        return "xacv"
    return None


def extract_dataset_name(root: str, input_root: str, runtime_config: Optional[Dict[str, object]] = None) -> str:
    input_dataset = canonical_dataset_name(os.path.basename(os.path.normpath(input_root)))
    if input_dataset is not None:
        return input_dataset

    rel_path = os.path.relpath(root, input_root)
    if rel_path == ".":
        root_dataset = canonical_dataset_name(os.path.basename(os.path.normpath(root)))
        if root_dataset is not None:
            return root_dataset
        return os.path.basename(os.path.normpath(root))

    rel_parts = os.path.normpath(rel_path).split(os.sep)
    for part in rel_parts:
        dataset_name = canonical_dataset_name(part)
        if dataset_name is not None:
            return dataset_name

    for part in rel_parts:
        if parse_scope_fold_count(part) is None:
            return part

    if runtime_config is not None:
        config_dataset = str(runtime_config.get("dataset", "")).strip()
        canonical_config_dataset = canonical_dataset_name(config_dataset)
        if canonical_config_dataset is not None:
            return canonical_config_dataset
        if config_dataset:
            return config_dataset

    return os.path.basename(os.path.normpath(input_root))


_SVAR_PROPOSED_TO_BRANCHES: Dict[str, str] = {
    "s13": "ip",
    "s14": "if",
    "s123": "imp",
    "s1234": "impf",
}


def normalize_proposed_suffix(name: str) -> str:
    """Replace _sXXX_proposed with the canonical fusion-branches abbreviation."""
    def _replace(m: re.Match) -> str:
        branches = _SVAR_PROPOSED_TO_BRANCHES.get(m.group(1))
        return f"_{branches}" if branches else "_imp"
    name = re.sub(r"_(s\d+)_proposed\b", _replace, name)
    return re.sub(r"_proposed\b", "_imp", name)


def parse_experiment_metadata(root_name: str) -> Dict[str, object]:
    root_name = normalize_proposed_suffix(root_name)
    tokens = root_name.split("_")
    loss = "unknown"
    conn_layout: object = "default"
    seg_aux_weight: Optional[float] = None
    seg_aux_variant = "none"

    # Group 1: numeric weight; group 2: primary suffix (no "-", fusion-branches candidate);
    # group 3: optional extra annotation allowing "-" and "." (e.g. "c5-0.3").
    segaux_match = re.search(
        r"(?<!drop)_segaux(?:_w([0-9]*\.?[0-9]+))?(?:_([A-Za-z][A-Za-z0-9]*)(?:_([A-Za-z0-9][A-Za-z0-9.-]*))?)?$",
        root_name,
    )
    segaux_suffix: Optional[str] = None
    if segaux_match is not None:
        segaux_weight_text = segaux_match.group(1)
        segaux_suffix = segaux_match.group(2)   # primary suffix: "io", "im", "imp", …
        segaux_extra  = segaux_match.group(3)   # extra annotation: "c5-0.3", "off", …

        if segaux_suffix and segaux_extra:
            _combined: Optional[str] = f"{segaux_suffix}_{segaux_extra}"
        elif segaux_suffix:
            _combined = segaux_suffix
        else:
            _combined = None

        if segaux_weight_text is not None:
            seg_aux_weight = float(segaux_weight_text)
            seg_aux_variant = (
                f"w{segaux_weight_text}({_combined})" if _combined
                else f"w{segaux_weight_text}"
            )
        else:
            seg_aux_variant = f"segaux({_combined})" if _combined else "segaux"

    conn_aux_c3_weight: Optional[float] = None
    conn_aux_c5_weight: Optional[float] = None

    # Fallback: _segaux_w<W>_c3-<C3>_c5-<C5>[_<branches>] (c3/c5 before branches)
    if segaux_match is None:
        _c3c5_m = re.search(
            r"(?<!drop)_segaux_w([0-9]*\.?[0-9]+)_c3-([0-9]+(?:\.[0-9]+)?)_c5-([0-9]+(?:\.[0-9]+)?)(?:_([A-Za-z][A-Za-z0-9]*))?$",
            root_name,
        )
        if _c3c5_m is not None:
            _wt = _c3c5_m.group(1)
            _c3v = _c3c5_m.group(2)
            _c5v = _c3c5_m.group(3)
            _fb = _c3c5_m.group(4)
            _ann = f"c3-{_c3v}_c5-{_c5v}"
            seg_aux_weight = float(_wt)
            seg_aux_variant = f"w{_wt}({_fb}_{_ann})" if _fb else f"w{_wt}({_ann})"
            segaux_suffix = _fb
            conn_aux_c3_weight = float(_c3v)
            conn_aux_c5_weight = float(_c5v)

    # c5 annotation (with or without segaux): _c5-<C5>[_<branches>] at end of name
    if conn_aux_c5_weight is None:
        _c5ns = re.search(r"_c5-([0-9]+(?:\.[0-9]+)?)(?:_[A-Za-z][A-Za-z0-9]*)?$", root_name)
        if _c5ns is not None:
            conn_aux_c5_weight = float(_c5ns.group(1))

    # Preferred fork naming convention:
    #   <label_mode>[_<fusion_tag>]_ <conn_num> [_<conn_layout>] _<loss> [_suffix...]
    # Example:
    #   dist_inverted_dg_8_gjml_sf_l1
    #   dist_inverted_dg_8_gjml_sf_l1_segaux
    ordered_name_match = re.fullmatch(
        r"(binary|dist|dist_inverted)(?:_(.+?))?_(\d+)(?:_(standard8|full24|out8|mid8|scale3_8|scale4_8))?_(bce|smooth_l1|cl_dice|gjml_sf_l1|gjml_sj_l1)(?:_.+)?",
        root_name,
    )
    if ordered_name_match is not None:
        ordered_label_mode = ordered_name_match.group(1)
        ordered_body = ordered_name_match.group(2)
        conn_num = int(ordered_name_match.group(3))
        explicit_layout = ordered_name_match.group(4)
        conn_layout = explicit_layout if explicit_layout is not None else "default"
        loss = ordered_name_match.group(5)
        if ordered_body:
            experiment = f"{ordered_label_mode}_{ordered_body}"
        else:
            experiment = ordered_label_mode
        tokens = experiment.split("_")
    else:
        experiment = root_name

    loss_suffixes = [
        (["gjml", "sf", "l1"], "gjml_sf_l1"),
        (["gjml", "sj", "l1"], "gjml_sj_l1"),
        (["smooth", "l1"], "smooth_l1"),
        (["cl", "dice"], "cl_dice"),
        (["bce"], "bce"),
    ]
    for suffix_tokens, loss_name in loss_suffixes:
        if tokens[-len(suffix_tokens):] == suffix_tokens:
            tokens = tokens[:-len(suffix_tokens)]
            loss = loss_name
            break

    conn_num: object = "NA" if ordered_name_match is None else int(ordered_name_match.group(3))
    if tokens:
        if len(tokens) >= 2 and tokens[-2].isdigit() and tokens[-1] in {"standard8", "full24", "out8", "mid8"}:
            conn_num = int(tokens[-2])
            conn_layout = tokens[-1]
            tokens = tokens[:-2]
        elif len(tokens) >= 3 and tokens[-3].isdigit() and tokens[-2] in {"scale3", "scale4"} and tokens[-1] == "8":
            conn_num = int(tokens[-3])
            conn_layout = f"{tokens[-2]}_8"
            tokens = tokens[:-3]
        elif tokens[-1].isdigit():
            conn_num = int(tokens[-1])
            tokens = tokens[:-1]

    if ordered_name_match is None:
        experiment = "_".join(tokens) if tokens else root_name
    if loss == "unknown":
        loss = infer_loss_tag_from_legacy_name(root_name)
    if experiment == "binary":
        loss = "bce"

    conn_fusion = "none"
    fusion_branches = "none"
    fusion_residual_scale: Optional[float] = None
    decoder_fusion = "none"
    label_mode = infer_label_mode_from_experiment(experiment)
    experiment_prefixes = ALLOWED_LABEL_MODES
    for prefix in experiment_prefixes:
        prefix_token = f"{prefix}_"
        if not experiment.startswith(prefix_token):
            continue
        experiment_body = experiment[len(prefix_token):]
        for fusion_name in KNOWN_CONN_FUSIONS:
            fusion_token = f"{fusion_name}_"
            if not experiment_body.startswith(fusion_token):
                continue
            fusion_tail = experiment_body[len(fusion_token):]
            parts = [p for p in fusion_tail.split("_") if p]
            if not parts:
                continue

            conn_fusion = normalize_conn_fusion_name(fusion_name)
            next_idx = 0
            # Skip optional single-letter loss-profile tag (A, B, C, …)
            if len(parts) > next_idx and re.fullmatch(r"[A-Z]", parts[next_idx]):
                next_idx += 1
            if len(parts) > next_idx:
                rs_match = re.fullmatch(r"rs([0-9]*\.?[0-9]+)", parts[next_idx])
                if rs_match is not None:
                    fusion_residual_scale = float(rs_match.group(1))
                    next_idx += 1
            if len(parts) > next_idx:
                fusion_branches_candidate = normalize_fusion_branches_name(parts[next_idx])
                if fusion_branches_candidate != "none":
                    fusion_branches = fusion_branches_candidate
                    next_idx += 1
            if len(parts) >= (next_idx + 2) and parts[next_idx] == "dec":
                decoder_fusion = "_".join(parts[next_idx:])
            break
        if conn_fusion != "none":
            break

    # If the segaux primary suffix is a known fusion-branches abbreviation and the
    # fusion was already identified but branches not yet set, promote it.
    if segaux_suffix is not None and conn_fusion != "none" and fusion_branches == "none":
        candidate = normalize_fusion_branches_name(segaux_suffix)
        if candidate != "none":
            fusion_branches = candidate

    # In the canonical name format (new), fusion_branches is embedded in the fusion tag rather
    # than appended after c3/c5 — so _fb was None when seg_aux_variant was built.  Backfill the
    # fusion_branches prefix into the variant string so _format_segaux_column_value can strip it
    # and produce the correct "w{W}/c3-{C3}_c5-{C5}" display.
    if (
        segaux_suffix is None
        and fusion_branches != "none"
        and seg_aux_variant != "none"
        and conn_aux_c3_weight is not None
    ):
        _m = re.fullmatch(r"w([0-9]*\.?[0-9]+)\((.+)\)", seg_aux_variant)
        if _m is not None and not _m.group(2).startswith(fusion_branches + "_"):
            seg_aux_variant = f"w{_m.group(1)}({fusion_branches}_{_m.group(2)})"

    return {
        "experiment": experiment,
        "label_mode": label_mode,
        "conn_num": conn_num,
        "conn_layout": conn_layout,
        "loss": loss,
        "conn_fusion": conn_fusion,
        "fusion_branches": fusion_branches,
        "fusion_residual_scale": fusion_residual_scale,
        "decoder_fusion": decoder_fusion,
        "seg_aux_weight": seg_aux_weight,
        "seg_aux_variant": seg_aux_variant,
        "conn_aux_c3_weight": conn_aux_c3_weight,
        "conn_aux_c5_weight": conn_aux_c5_weight,
    }


def extract_comparison_model_name(root: str, input_root: str) -> Optional[str]:
    """Returns the top-level directory name when it is a comparison model, else None.

    DconnNet experiments live under <input_root>/<dataset>/<experiment>/, so
    their first path component is a known dataset name.  Comparison models live
    under <input_root>/<model_name>/<dataset>/, so their first component is
    NOT a dataset name.
    """
    rel_path = os.path.relpath(root, input_root)
    if rel_path == ".":
        return None
    rel_parts = os.path.normpath(rel_path).split(os.sep)
    first_part = rel_parts[0]
    if parse_scope_fold_count(first_part) is not None:
        return None
    normalized = _normalize_dataset_name(first_part)
    if (
        canonical_dataset_name(first_part) is not None
        or normalized == "deepglobe"
        or normalized == "deepcrack"
        or is_octa_dataset_name(normalized)
    ):
        return None
    return first_part


def load_runtime_config(root: str) -> Optional[Dict[str, object]]:
    config_path = os.path.join(root, "config.yaml")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _normalize_config_conn_layout(conn_num: object, conn_layout: object) -> str:
    layout_name = str(conn_layout if conn_layout is not None else "").strip().lower()
    if layout_name in {"", "default"}:
        return "default"

    try:
        conn_num_int = int(conn_num)
    except (TypeError, ValueError):
        return layout_name

    if (conn_num_int, layout_name) in {(8, "standard8"), (24, "full24")}:
        return "default"
    return layout_name


def build_experiment_name_from_runtime_config(config: Dict[str, object]) -> Optional[str]:
    label_mode = str(config.get("label_mode", "")).strip()
    if label_mode not in ALLOWED_LABEL_MODES:
        return None

    try:
        conn_num = int(config.get("conn_num"))
    except (TypeError, ValueError):
        return None

    conn_layout = _normalize_config_conn_layout(conn_num, config.get("conn_layout", "default"))
    layout_suffix = "" if conn_layout == "default" else f"_{conn_layout}"

    conn_fusion = normalize_conn_fusion_name(config.get("conn_fusion", "none"))
    fusion_branches = normalize_fusion_branches_name(config.get("fusion_branches", "io"))
    dist_aux_loss = str(config.get("dist_aux_loss", "unknown")).strip() or "unknown"

    if conn_fusion == "none":
        if label_mode == "binary":
            experiment_name = f"binary_{conn_num}{layout_suffix}_bce"
        else:
            experiment_name = f"{label_mode}_{conn_num}{layout_suffix}_{dist_aux_loss}"
    else:
        fusion_tag = conn_fusion
        if conn_fusion == "scaled_sum":
            try:
                residual_scale = float(config.get("fusion_residual_scale", 0.2))
                scale_str = f"{residual_scale:.6f}".rstrip("0").rstrip(".")
                fusion_tag = f"{fusion_tag}_rs{scale_str}"
            except (TypeError, ValueError):
                pass
        if fusion_branches != "none":
            fusion_tag = f"{fusion_tag}_{fusion_branches}"
        if label_mode == "binary":
            experiment_name = f"binary_{fusion_tag}_{conn_num}{layout_suffix}_bce"
        else:
            experiment_name = f"{label_mode}_{fusion_tag}_{conn_num}{layout_suffix}_{dist_aux_loss}"

    seg_aux_weight = config.get("seg_aux_weight")
    use_seg_aux = bool(config.get("use_seg_aux", False))
    if use_seg_aux:
        if seg_aux_weight is not None:
            experiment_name = f"{experiment_name}_segaux_w{seg_aux_weight}"
        else:
            experiment_name = f"{experiment_name}_segaux"

    experiment_name_suffix = str(config.get("experiment_name_suffix", "") or "").strip()
    # Avoid duplicating fusion_branches when it is already embedded in the fusion tag.
    # Also strip a trailing _<fusion_branches> token from the suffix (e.g. "c5-0.2_io" → "c5-0.2")
    # because older runs appended fusion_branches as the last element of experiment_name_suffix
    # instead of embedding it in the fusion tag.
    if fusion_branches != "none":
        if experiment_name_suffix == fusion_branches:
            experiment_name_suffix = ""
        elif experiment_name_suffix.endswith(f"_{fusion_branches}"):
            experiment_name_suffix = experiment_name_suffix[: -(len(fusion_branches) + 1)]
    if experiment_name_suffix:
        experiment_name = f"{experiment_name}_{experiment_name_suffix}"

    return experiment_name


def _overlay_segaux_from_config(
    meta: Dict[str, object],
    runtime_config: Dict[str, object],
) -> Dict[str, object]:
    """Fix seg_aux_weight/variant from config when regex failed to parse them.

    The _c3-X_c5-Y_io suffix format can defeat the segaux regex, so this reads
    seg_aux_weight directly from config as a fallback without touching c3/c5 fields
    (those are set by parse_experiment_metadata via directory-name regex).
    """
    if meta.get("seg_aux_weight") is not None:
        return meta
    if not runtime_config.get("use_seg_aux"):
        return meta
    saw = runtime_config.get("seg_aux_weight")
    if saw is None:
        return meta
    try:
        saw_f = float(saw)
    except (TypeError, ValueError):
        return meta
    out = dict(meta)
    out["seg_aux_weight"] = saw_f
    if str(out.get("seg_aux_variant", "none")) == "none":
        fb = str(out.get("fusion_branches", "none"))
        c3v = out.get("conn_aux_c3_weight")
        c5v = out.get("conn_aux_c5_weight")
        wt = f"{saw_f:g}"
        if c3v is not None or c5v is not None:
            ann_parts: List[str] = []
            if c3v is not None:
                ann_parts.append(f"c3-{float(c3v):g}")
            if c5v is not None:
                ann_parts.append(f"c5-{float(c5v):g}")
            ann = "_".join(ann_parts)
            out["seg_aux_variant"] = (
                f"w{wt}({fb}_{ann})" if fb not in {"none"} else f"w{wt}({ann})"
            )
        else:
            out["seg_aux_variant"] = f"w{wt}({fb})" if fb not in {"none"} else f"w{wt}"
    return out


def resolve_experiment_metadata(root: str) -> Dict[str, object]:
    runtime_config = load_runtime_config(root)
    if runtime_config is not None:
        config_experiment_name = build_experiment_name_from_runtime_config(runtime_config)
        if config_experiment_name is not None:
            # Forward a c5-X.X annotation present in the folder name but absent from
            # the config-derived canonical name (e.g. when experiment_name_suffix is
            # just "io" but the folder was created with "_io_c5-0.2").
            root_name = os.path.basename(os.path.normpath(root))
            c5_match = re.search(r"_(c5-[0-9]+(?:\.[0-9]+)?)$", root_name)
            if c5_match is not None and "c5-" not in config_experiment_name:
                c5_token = c5_match.group(1)
                fb = normalize_fusion_branches_name(runtime_config.get("fusion_branches", "none"))
                if fb not in {"none", "io"}:
                    config_experiment_name = f"{config_experiment_name}_{fb}_{c5_token}"
                else:
                    config_experiment_name = f"{config_experiment_name}_io_{c5_token}"
            meta = parse_experiment_metadata(config_experiment_name)
            meta = _overlay_segaux_from_config(meta, runtime_config)
            # Overlay c3/c5 from runtime config when not extractable from the name.
            # Handles experiments where experiment_name_suffix uses the old 'c3w{x}_c5w{y}'
            # format while the directory name uses 'c3-{x}_c5-{y}'.
            if meta.get("conn_aux_c3_weight") is None:
                c3_cfg = runtime_config.get("conn_aux_c3_weight")
                if c3_cfg is not None:
                    try:
                        meta = dict(meta)
                        meta["conn_aux_c3_weight"] = float(c3_cfg)
                    except (TypeError, ValueError):
                        pass
            if meta.get("conn_aux_c5_weight") is None:
                c5_cfg = runtime_config.get("conn_aux_c5_weight")
                if c5_cfg is not None:
                    try:
                        meta = dict(meta)
                        meta["conn_aux_c5_weight"] = float(c5_cfg)
                    except (TypeError, ValueError):
                        pass
            return meta
    root_name = os.path.basename(os.path.normpath(root))
    return parse_experiment_metadata(root_name)


def infer_label_mode_from_experiment(experiment_name: object) -> str:
    experiment = str(experiment_name)
    if experiment == "binary" or experiment.startswith("binary_"):
        return "binary"
    if experiment == "dist_inverted" or experiment.startswith("dist_inverted_"):
        return "dist_inverted"
    if experiment == "dist" or experiment.startswith("dist_"):
        return "dist"
    return "unknown"


def _format_fusion_display_name(conn_fusion: object, fusion_branches: object = "none") -> str:
    conn_fusion_name = normalize_conn_fusion_name(conn_fusion)
    fusion_branches_name = normalize_fusion_branches_name(fusion_branches)
    if conn_fusion_name == "none":
        return "none"
    if fusion_branches_name == "none":
        return conn_fusion_name
    return f"{conn_fusion_name}/{fusion_branches_name}"


def _append_decoder_variant_to_fusion_label(label: str, decoder_fusion: object = "none") -> str:
    decoder_variant = normalize_decoder_fusion_name(decoder_fusion)
    if decoder_variant == "none":
        return label
    return f"{label}/{decoder_variant}"


def _format_fusion_table_label(
    conn_fusion: object,
    fusion_residual_scale: object = None,
    fusion_branches: object = "none",
    decoder_fusion: object = "none",
) -> str:
    conn_fusion_label = _format_fusion_display_name(conn_fusion, fusion_branches)
    conn_fusion_name = normalize_conn_fusion_name(conn_fusion)
    if conn_fusion_name == "none":
        return "none"
    if conn_fusion_name == "scaled_sum" and fusion_residual_scale is not None:
        try:
            return _append_decoder_variant_to_fusion_label(
                f"{conn_fusion_label}/rs{float(fusion_residual_scale):g}",
                decoder_fusion,
            )
        except (TypeError, ValueError):
            pass
    return _append_decoder_variant_to_fusion_label(
        conn_fusion_label,
        decoder_fusion,
    )


def _format_seg_aux_table_label(seg_aux_weight: object) -> Optional[str]:
    if seg_aux_weight is None:
        return None
    if isinstance(seg_aux_weight, (int, float)):
        return f"{float(seg_aux_weight):g}"
    text = str(seg_aux_weight).strip()
    return text if text else None


def _format_segaux_column_value(
    seg_aux_weight: object,
    seg_aux_variant: object,
    fusion_branches: object = "none",
) -> Optional[str]:
    seg_aux_label = _format_seg_aux_table_label(seg_aux_weight)
    seg_aux_variant_name = str(seg_aux_variant if seg_aux_variant is not None else "none")
    fusion_branches_name = normalize_fusion_branches_name(fusion_branches)
    if seg_aux_variant_name == "segaux" and seg_aux_label is None:
        return "segaux"
    if seg_aux_variant_name.startswith("segaux(") and seg_aux_label is None:
        match = re.fullmatch(r"segaux\(([^)]+)\)", seg_aux_variant_name)
        if (
            match is not None
            and fusion_branches_name not in {"none"}
            and match.group(1) == fusion_branches_name
        ):
            return "segaux"
        return seg_aux_variant_name
    if seg_aux_label is None:
        return None
    if seg_aux_variant_name == "segaux":
        # SegAux is enabled with an explicit numeric weight.
        return seg_aux_label
    if seg_aux_variant_name.startswith("w"):
        match = re.fullmatch(r"w([0-9]*\.?[0-9]+)\(([^)]+)\)", seg_aux_variant_name)
        if match is not None and fusion_branches_name not in {"none"}:
            inner = match.group(2)
            if inner == fusion_branches_name:
                return seg_aux_label
            if inner.startswith(fusion_branches_name + "_"):
                remaining = inner[len(fusion_branches_name) + 1:]
                return f"w{seg_aux_label}/{remaining}"
        return seg_aux_variant_name
    return seg_aux_label


def _table_fusion_and_decoder_labels(
    conn_fusion: object,
    fusion_residual_scale: object,
    fusion_branches: object = "none",
    decoder_fusion: object = "none",
    seg_aux_weight: object = None,
    seg_aux_variant: object = "none",
) -> Tuple[str, str]:
    conn_fusion_name = normalize_conn_fusion_name(conn_fusion)
    seg_aux_label = _format_segaux_column_value(seg_aux_weight, seg_aux_variant, fusion_branches)

    if seg_aux_label is not None:
        return _format_fusion_table_label(
            conn_fusion_name,
            fusion_residual_scale,
            fusion_branches,
            decoder_fusion,
        ), seg_aux_label

    return (
        _format_fusion_table_label(
            conn_fusion_name,
            fusion_residual_scale,
            fusion_branches,
            decoder_fusion,
        ),
        "none",
    )


def _latex_summary_fusion_and_decoder_labels(
    conn_fusion: object,
    fusion_residual_scale: object,
    fusion_branches: object = "none",
    decoder_fusion: object = "none",
    seg_aux_weight: object = None,
    seg_aux_variant: object = "none",
) -> Tuple[str, str]:
    conn_fusion_name = normalize_conn_fusion_name(conn_fusion)
    conn_fusion_label = _append_decoder_variant_to_fusion_label(
        _format_fusion_display_name(conn_fusion_name, fusion_branches),
        decoder_fusion,
    )

    if conn_fusion_name == "scaled_sum":
        seg_aux_label = _format_segaux_column_value(seg_aux_weight, seg_aux_variant, fusion_branches)
        dec_label = "none"
        if seg_aux_label is not None:
            dec_label = seg_aux_label
        if fusion_residual_scale is not None:
            try:
                return f"{conn_fusion_label}/rs{float(fusion_residual_scale):g}", dec_label
            except (TypeError, ValueError):
                pass
        return conn_fusion_label, dec_label

    _, decoder_label = _table_fusion_and_decoder_labels(
        conn_fusion,
        fusion_residual_scale,
        fusion_branches,
        decoder_fusion,
        seg_aux_weight,
        seg_aux_variant,
    )
    return conn_fusion_label, decoder_label


def _latex_ablation_fusion_spec(
    conn_fusion: object,
    fusion_residual_scale: object = None,
    decoder_fusion: object = "none",
) -> str:
    conn_fusion_name = normalize_conn_fusion_name(conn_fusion)
    decoder_variant = normalize_decoder_fusion_name(decoder_fusion)
    if conn_fusion_name in {"dg", "dg_direct"}:
        return "" if decoder_variant == "none" else decoder_variant
    if conn_fusion_name == "scaled_sum":
        if fusion_residual_scale is not None:
            try:
                base_spec = f"rs{float(fusion_residual_scale):g}"
                return base_spec if decoder_variant == "none" else f"{base_spec}/{decoder_variant}"
            except (TypeError, ValueError):
                return "" if decoder_variant == "none" else decoder_variant
        return "" if decoder_variant == "none" else decoder_variant
    return "" if decoder_variant == "none" else decoder_variant


def _conn_fusion_sort_value(conn_fusion_name: object) -> int:
    conn_fusion = normalize_conn_fusion_name(conn_fusion_name)
    if conn_fusion not in CONN_FUSION_SORT_ORDER and "/" in str(conn_fusion_name):
        conn_fusion = normalize_conn_fusion_name(str(conn_fusion_name).rsplit("/", 1)[-1])
    return CONN_FUSION_SORT_ORDER.get(conn_fusion, 9)


def _loss_sort_key(loss_name: object) -> Tuple[int, str]:
    loss = str(loss_name if loss_name is not None else "unknown")
    loss_order = {
        "bce": 0,
        "smooth_l1": 1,
        "cl_dice": 2,
        "gjml_sf_l1": 3,
        "gjml_sj_l1": 3,
        "unknown": 9,
    }.get(loss, 9)
    return (loss_order, loss.lower())


def _logical_experiment_key(row: Dict[str, object]) -> Tuple[object, ...]:
    """Key that identifies an experiment by logical configuration, ignoring folder name.

    Used for cross-folder deduplication: when two folders parse to identical logical
    configuration, the one from the most recently modified directory wins.
    """
    try:
        weight = float(row.get("seg_aux_weight") or 0)
    except (TypeError, ValueError):
        weight = 0.0
    c3_w = row.get("conn_aux_c3_weight")
    c5_w = row.get("conn_aux_c5_weight")
    return (
        str(row.get("dataset", "")),
        str(row.get("label_mode", "")),
        row.get("conn_num", "NA"),
        str(row.get("conn_layout", "default")),
        str(row.get("conn_fusion", "none")),
        str(row.get("fusion_branches", "none")),
        str(row.get("fusion_loss_profile", "A")),
        row.get("fusion_residual_scale"),
        str(row.get("loss", "")),
        round(weight, 6),
        str(row.get("seg_aux_variant", "none")),
        str(row.get("fold_scope", "")),
        round(float(c3_w), 6) if c3_w is not None else None,
        round(float(c5_w), 6) if c5_w is not None else None,
    )


def _experiment_mean_unique_key(row: Dict[str, object]) -> Tuple[object, ...]:
    # Keep one row per experiment directory name so suffix variants
    # (for example *_segaux, *_segaux_w0.1) are not collapsed.
    experiment_source = str(row.get("experiment_source", row.get("experiment", "unknown")))
    return (
        row["dataset"],
        experiment_source,
        row["experiment"],
        row.get("label_mode", "unknown"),
        row["conn_num"],
        row.get("conn_layout", "default"),
        row.get("conn_fusion", "none"),
        row.get("fusion_branches", "none"),
        row.get("decoder_fusion", "none"),
        row.get("seg_aux_variant", "none"),
        row["loss"],
        row["fold_scope"],
    )


def experiment_sort_key(row: Dict[str, object]) -> Tuple[object, ...]:
    label_mode = str(row.get("label_mode", infer_label_mode_from_experiment(row.get("experiment", "unknown"))))
    label_mode_order = {
        "binary": 0,
        "dist": 1,
        "dist_inverted": 2,
    }.get(label_mode, 9)

    conn_raw = row.get("conn_num", "NA")
    conn_sort = int(conn_raw) if isinstance(conn_raw, int) else 10 ** 9
    conn_layout_sort = _conn_layout_sort_key(row.get("conn_layout", "default"))

    fusion_label, decoder_label = _table_fusion_and_decoder_labels(
        row.get("conn_fusion", "none"),
        row.get("fusion_residual_scale"),
        row.get("fusion_branches", "none"),
        row.get("decoder_fusion", "none"),
        row.get("seg_aux_weight"),
        row.get("seg_aux_variant", "none"),
    )
    fusion_sort = _conn_fusion_sort_value(row.get("conn_fusion", "none"))
    decoder_sort = segaux_sort_key(decoder_label)

    loss_order, loss_norm = _loss_sort_key(row.get("loss", "Unknown"))

    fold_scope = str(row.get("fold_scope", "direct"))
    fold_scope_count = row.get("fold_scope_count", "NA")
    fold_scope_sort = int(fold_scope_count) if isinstance(fold_scope_count, int) else 10 ** 9

    return (
        label_mode_order,
        natural_sort_key(label_mode),
        conn_sort,
        conn_layout_sort,
        fusion_sort,
        natural_sort_key(str(row.get("fusion_branches", "none"))),
        natural_sort_key(fusion_label),
        decoder_sort,
        loss_order,
        loss_norm,
        fold_scope_sort,
        fold_scope.lower(),
    )


def write_experiment_mean_csv(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "experiment_source",
                "label_mode",
                "conn_num",
                "conn_label",
                "conn_layout",
                "conn_fusion",
                "fusion_branches",
                "fusion_residual_scale",
                "loss",
                "num_folds",
                "best_dice_mean",
                "best_dice_std",
                "best_jac_mean",
                "best_jac_std",
                "best_cldice_mean",
                "best_cldice_std",
                "best_precision_mean",
                "best_precision_std",
                "best_accuracy_mean",
                "best_accuracy_std",
                "best_betti_error_0_mean",
                "best_betti_error_0_std",
                "best_betti_error_1_mean",
                "best_betti_error_1_std",
                "best_hd95_mean",
                "best_hd95_std",
                "train_elapsed_mean_hms",
                "train_elapsed_std_hms",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["dataset"],
                    row.get("experiment_source", "NA"),
                    str(row.get("label_mode", infer_label_mode_from_experiment(row.get("experiment", "unknown")))),
                    row["conn_num"],
                    _format_conn_table_label(row["conn_num"], row.get("conn_layout", "default")),
                    row.get("conn_layout", "default"),
                    row.get("conn_fusion", "none"),
                    row.get("fusion_branches", "none"),
                    _fmt_float(
                        (
                            float(row.get("fusion_residual_scale"))
                            if row.get("fusion_residual_scale") is not None
                            else float("nan")
                        )
                    ),
                    row["loss"],
                    int(row["num_folds"]),
                    _fmt_float(row["best_dice"]),
                    _fmt_float(row.get("best_dice_std", float("nan"))),
                    _fmt_float(row["best_jac"]),
                    _fmt_float(row.get("best_jac_std", float("nan"))),
                    _fmt_float(row["best_cldice"]),
                    _fmt_float(row.get("best_cldice_std", float("nan"))),
                    _fmt_float(row.get("best_precision", float("nan"))),
                    _fmt_float(row.get("best_precision_std", float("nan"))),
                    _fmt_float(row.get("best_accuracy", float("nan"))),
                    _fmt_float(row.get("best_accuracy_std", float("nan"))),
                    _fmt_float(row["best_betti_error_0"]),
                    _fmt_float(row.get("best_betti_error_0_std", float("nan"))),
                    _fmt_float(row["best_betti_error_1"]),
                    _fmt_float(row.get("best_betti_error_1_std", float("nan"))),
                    _fmt_float(float(row.get("best_hd95", float("nan")))),
                    _fmt_float(float(row.get("best_hd95_std", float("nan")))),
                    _fmt_duration_csv(row.get("train_elapsed_hms", "")),
                    _fmt_duration_csv(row.get("train_elapsed_std_hms", "")),
                ]
            )


def write_latex(
    path: str,
    title: str,
    fold_rows: List[Dict[str, object]],
    mean_row: Dict[str, object],
    std_row: Dict[str, object],
    dataset_name: str,
) -> None:
    metric_specs = dataset_fold_metric_specs(dataset_name)
    table_columns = 3 + len(metric_specs)
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a2paper, landscape, margin=1in]{geometry}",
        r"\usepackage{amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{float}",
        r"\usepackage{multicol}",
        r"\usepackage{caption}",
        r"\usepackage{multirow}",
        r"\captionsetup[table]{font=scriptsize}",
        r"\begin{document}",
        rf"\renewcommand{{\arraystretch}}{{{LATEX_TABLE_ARRAYSTRETCH}}}",
        rf"\section*{{{title}}}",
        r"\begin{table}[H]",
        r"\centering",
        rf"\begin{{tabular}}{{{'l' + 'c' * (table_columns - 1)}}}",
        r"\toprule",
        "Fold & Best Epoch & " + " & ".join(label for _, label in metric_specs) + r" & Train Time \\",
        r"\midrule",
    ]

    for row in fold_rows:
        time_txt = _fmt_duration_latex(row.get("train_elapsed_hms", ""))
        metric_vals = []
        for key, _ in metric_specs:
            metric_vals.append(_fmt_float(float(row.get(key, float("nan")))))
        lines.append(
            f'{int(row["fold"])} & {int(row["best_epoch"])} & '
            + " & ".join(metric_vals)
            + f' & {time_txt} \\\\'
        )

    mean_metrics = [_fmt_float(float(mean_row.get(key, float("nan")))) for key, _ in metric_specs]
    std_metrics = [_fmt_float(float(std_row.get(key, float("nan")))) for key, _ in metric_specs]
    mean_time = _fmt_duration_latex(mean_row.get("train_elapsed_hms", ""))
    std_time = _fmt_duration_latex(std_row.get("train_elapsed_hms", ""))
    lines += [
        r"\midrule",
        f'Mean & {mean_row["best_epoch"]:.4f} & ' + " & ".join(mean_metrics) + f' & {mean_time} \\\\',
        f'Std & {std_row["best_epoch"]:.4f} & ' + " & ".join(std_metrics) + f' & {std_time} \\\\',
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.15cm}",
        r"\caption{final result summary}",
        r"\end{table}",
        r"\end{document}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _append_experiment_mean_table_lines(lines: List[str], rows: List[Dict[str, object]], caption: str) -> None:
    dice_best, dice_second = _rank_row_indices(rows, "best_dice")
    jac_best, jac_second = _rank_row_indices(rows, "best_jac")
    cldice_best, cldice_second = _rank_row_indices(rows, "best_cldice")
    betti0_best, betti0_second = _rank_row_indices(rows, "best_betti_error_0", higher_is_better=False)
    betti1_best, betti1_second = _rank_row_indices(rows, "best_betti_error_1", higher_is_better=False)
    hd95_best, hd95_second = _rank_row_indices(rows, "best_hd95", higher_is_better=False)

    lines += [
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{lccccccccccccccccccccc}",
        r"\toprule",
        r"No. & label\_mode & Conn & Fusion & SegAux & Loss & Dice $\uparrow$ & \textcolor{gray}{(std)} & Jac $\uparrow$ & \textcolor{gray}{(std)} & clDice $\uparrow$ & \textcolor{gray}{(std)} & Err $(\beta_0)$ $\downarrow$ & \textcolor{gray}{(std)} & Err $(\beta_1)$ $\downarrow$ & \textcolor{gray}{(std)} & HD95 $\downarrow$ & \textcolor{gray}{(std)} & Train Time & \textcolor{gray}{(std)} \\",
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        fusion_label, decoder_label = _latex_summary_fusion_and_decoder_labels(
            row.get("conn_fusion", "none"),
            row.get("fusion_residual_scale"),
            row.get("fusion_branches", "none"),
            row.get("decoder_fusion", "none"),
            row.get("seg_aux_weight"),
            row.get("seg_aux_variant", "none"),
        )
        dice_mean, dice_std = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_dice"]),
            float(row.get("best_dice_std", float("nan"))),
            idx in dice_best,
            idx in dice_second,
        )
        jac_mean, jac_std = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_jac"]),
            float(row.get("best_jac_std", float("nan"))),
            idx in jac_best,
            idx in jac_second,
        )
        cldice_mean, cldice_std = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_cldice"]),
            float(row.get("best_cldice_std", float("nan"))),
            idx in cldice_best,
            idx in cldice_second,
        )
        betti0_mean, betti0_std = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_betti_error_0"]),
            float(row.get("best_betti_error_0_std", float("nan"))),
            idx in betti0_best,
            idx in betti0_second,
        )
        betti1_mean, betti1_std = _fmt_latex_ranked_mean_std_marginal(
            float(row["best_betti_error_1"]),
            float(row.get("best_betti_error_1_std", float("nan"))),
            idx in betti1_best,
            idx in betti1_second,
        )
        hd95_mean, hd95_std = _fmt_latex_ranked_mean_std_marginal(
            float(row.get("best_hd95", float("nan"))),
            float(row.get("best_hd95_std", float("nan"))),
            idx in hd95_best,
            idx in hd95_second,
        )
        time_mean = _fmt_duration_latex(row.get("train_elapsed_hms", ""))
        time_std = _fmt_duration_latex(row.get("train_elapsed_std_hms", ""))

        lines.append(
            f'{idx + 1} '
            f'& {escape_latex_text(str(row.get("label_mode", infer_label_mode_from_experiment(row.get("experiment", "unknown")))))} '
            f'& {escape_latex_text(_format_conn_table_label(row["conn_num"], row.get("conn_layout", "default")))} '
            f'& {escape_latex_text(fusion_label)} '
            f'& {escape_latex_text(decoder_label)} '
            f'& {escape_latex_text(str(row["loss"]))} '
            f'& {dice_mean} & {dice_std} '
            f'& {jac_mean} & {jac_std} '
            f'& {cldice_mean} & {cldice_std} '
            f'& {betti0_mean} & {betti0_std} '
            f'& {betti1_mean} & {betti1_std} '
            f'& {hd95_mean} & {hd95_std} '
            f'& {time_mean} & {time_std} \\\\'
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.15cm}",
        rf"\caption{{{caption}}}",
        r"\end{table}",
    ]


def dataset_metric_specs(dataset_name: str) -> List[Tuple[str, str, bool]]:
    name = dataset_name.lower()
    if name in {"chase", "drive"}:
        return [
            ("best_dice", "Dice", True),
            ("best_jac", "IoU", True),
            ("best_cldice", "clDice", True),
            ("best_betti_error_0", "B0 error", False),
            ("best_betti_error_1", "B1 error", False),
            ("best_hd95", "HD95", False),
        ]
    if name in {"isic", "cremi"}:
        return [
            ("best_dice", "Dice", True),
            ("best_jac", "IoU", True),
            ("best_accuracy", "Accuracy", True),
            ("best_precision", "Precision", True),
            ("best_hd95", "HD95", False),
        ]

    return [
        ("best_dice", "Dice", True),
        ("best_jac", "IoU", True),
        ("best_cldice", "clDice", True),
        ("best_betti_error_0", "B0 error", False),
        ("best_betti_error_1", "B1 error", False),
        ("best_hd95", "HD95", False),
    ]


def dataset_fold_metric_specs(dataset_name: str) -> List[Tuple[str, str]]:
    name = dataset_name.lower()
    if name in {"isic", "cremi"}:
        return [
            ("best_dice", "Dice"),
            ("best_jac", "IoU"),
            ("best_accuracy", "Accuracy"),
            ("best_precision", "Precision"),
            ("best_hd95", "HD95"),
        ]

    return [
        ("best_dice", "Dice"),
        ("best_jac", "IoU"),
        ("best_cldice", "clDice"),
        ("best_betti_error_0", "B0 error"),
        ("best_betti_error_1", "B1 error"),
        ("best_hd95", "HD95"),
    ]


_EXTERNAL_BASELINE_METHODS: List[Tuple[str, str]] = [
    ("U-Net",           "unet"),
    ("Attention U-Net", "attention_unet"),
    ("FRU-Net",         "frunet"),
    ("CSNet",           "csnet"),
    ("CAPE",            "CAPE"),
    ("SegDiff",         "segdiff"),
    ("MedSegDiff",      "medsegdiff"),
]

_EXTERNAL_BASELINE_METHODS_BY_DATASET: Dict[str, List[Tuple[str, str]]] = {
    "drive": _EXTERNAL_BASELINE_METHODS,
    "deepcrack": _EXTERNAL_BASELINE_METHODS,
}


def _dataset_baseline_dir_name(dataset_name: str) -> Optional[str]:
    n = _normalize_dataset_name(dataset_name)
    if n == "drive":
        return "drive"
    if n == "deepcrack":
        return "deepcrack"
    if n in {"chase", "chasedb1"}:
        return "chase"
    if n == "octa5003m":
        return "octa500-3M"
    if n == "octa5006m":
        return "octa500-6M"
    return None


def _load_external_baselines_for_dataset(
    dataset_name: str, baselines_root: str = "output",
) -> List[Tuple[str, Dict[str, float]]]:
    ds_dir = _dataset_baseline_dir_name(dataset_name)
    if ds_dir is None:
        return []
    out: List[Tuple[str, Dict[str, float]]] = []
    raw_keys = ("dice", "jac", "cldice", "betti_error_0", "betti_error_1", "hd95")
    methods = _EXTERNAL_BASELINE_METHODS_BY_DATASET.get(ds_dir, _EXTERNAL_BASELINE_METHODS)
    for label, subdir in methods:
        for fname in ("final_results.csv", "final_results_1.csv", "results.csv"):
            path = os.path.join(baselines_root, subdir, ds_dir, fname)
            if not os.path.isfile(path):
                continue
            with open(path, newline="") as f:
                row = next(csv.DictReader(f), None)
            if row is None:
                continue
            metrics = {
                f"best_{k}": _to_float(str(row.get(k, "nan"))) for k in raw_keys
            }
            out.append((label, metrics))
            break
    return out


def _append_external_baseline_table_lines(
    lines: List[str], dataset_name: str, start_num: int = 1,
) -> int:
    """Emit a baseline table for `dataset_name`. Returns count of rows emitted."""
    baselines = _load_external_baselines_for_dataset(dataset_name)
    if not baselines:
        return 0
    metric_specs = dataset_metric_specs(dataset_name)
    metric_dicts: List[Dict[str, object]] = [dict(m) for _, m in baselines]
    rank_map = {
        key: _rank_row_indices(metric_dicts, key, higher_is_better=higher)
        for key, _, higher in metric_specs
    }

    lines += [
        r"\begin{table}[H]",
        r"\centering",
        rf"\begin{{tabular}}{{{'l l' + 'c' * len(metric_specs)}}}",
        r"\toprule",
        " & ".join(["No.", "Method"] + [label for _, label, _ in metric_specs]) + r" \\",
        r"\midrule",
    ]
    for idx, (label, metrics) in enumerate(baselines):
        cells = [str(start_num + idx), escape_latex_text(label)]
        for key, _, _ in metric_specs:
            best_idx, second_idx = rank_map[key]
            cells.append(
                _fmt_latex_ranked_value(
                    float(metrics.get(key, float("nan"))),
                    idx in best_idx, idx in second_idx,
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.15cm}",
        rf"\caption{{External baselines ({escape_latex_text(dataset_name)})}}",
        r"\end{table}",
    ]
    return len(baselines)


def _build_tabular_lines(
    rows: List[Dict[str, object]],
    dataset_name: str,
    include_std: bool,
    include_loss: bool,
    start_num: int,
    rank_map: Dict[str, Tuple[Set[int], Set[int]]],
    rank_offset: int = 0,
    force_has_c3c5: Optional[bool] = None,
) -> List[str]:
    """Return \\begin{tabular}...\\end{tabular} lines without any surrounding table environment.

    rank_map keys map to absolute row indices across all rows (before splitting).
    rank_offset is the absolute index of the first row in this slice.
    force_has_c3c5: when provided, overrides per-slice auto-detection so both halves
    of a two-column split always have consistent column counts.
    """
    metric_specs = dataset_metric_specs(dataset_name)
    metric_col_count = 2 * len(metric_specs) if include_std else len(metric_specs)
    # +2 for separate c3/c5 columns (shown only when any row has them)
    if force_has_c3c5 is not None:
        has_c3c5 = force_has_c3c5
    else:
        has_c3c5 = True
    extra_cols = 2 if has_c3c5 else 0
    base_cols = (6 if include_loss else 5)
    table_columns = base_cols + extra_cols + metric_col_count
    header_parts = ["No.", "label\\_mode", "Conn", "Fusion", "SegAux"]
    if has_c3c5:
        header_parts.extend(["c3", "c5"])
    if include_loss:
        header_parts.append("Loss")
    if include_std:
        for _, label, _ in metric_specs:
            header_parts.extend([label, r"\textcolor{gray}{(std)}"])
    else:
        header_parts.extend(label for _, label, _ in metric_specs)

    result: List[str] = [
        rf"\begin{{tabular}}{{{'l' + 'c' * (table_columns - 1)}}}",
        r"\toprule",
        " & ".join(header_parts) + r" \\",
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        abs_idx = rank_offset + idx
        fusion_label, decoder_label = _latex_summary_fusion_and_decoder_labels(
            row.get("conn_fusion", "none"),
            row.get("fusion_residual_scale"),
            row.get("fusion_branches", "none"),
            row.get("decoder_fusion", "none"),
            row.get("seg_aux_weight"),
            row.get("seg_aux_variant", "none"),
        )

        # Split SegAux into base weight + separate c3/c5 columns
        c3_w = row.get("conn_aux_c3_weight")
        c5_w = row.get("conn_aux_c5_weight")
        if c3_w is not None or c5_w is not None:
            saw = row.get("seg_aux_weight")
            segaux_cell = _fmt_aux_weight(float(saw)) if saw is not None else "none"
            c3_cell = _fmt_aux_weight(float(c3_w)) if c3_w is not None else "--"
            c5_cell = _fmt_aux_weight(float(c5_w)) if c5_w is not None else "--"
        else:
            segaux_cell = decoder_label  # combined display (connaux_off etc.)
            c3_cell = "--"
            c5_cell = "--"

        metric_cells: List[str] = []
        for key, _, _ in metric_specs:
            best_indices, second_indices = rank_map[key]
            if include_std:
                mean_txt, std_txt = _fmt_latex_ranked_mean_std_marginal(
                    float(row.get(key, float("nan"))),
                    float(row.get(f"{key}_std", float("nan"))),
                    abs_idx in best_indices,
                    abs_idx in second_indices,
                )
                metric_cells.extend([mean_txt, std_txt])
            else:
                metric_cells.append(
                    _fmt_latex_ranked_value(
                        float(row.get(key, float("nan"))),
                        abs_idx in best_indices,
                        abs_idx in second_indices,
                    )
                )

        row_parts = [
            f"{start_num + idx}",
            escape_latex_text(str(row.get("label_mode", infer_label_mode_from_experiment(row.get("experiment", "unknown"))))),
            escape_latex_text(_format_conn_table_label(row["conn_num"], row.get("conn_layout", "default"))),
            escape_latex_text(fusion_label),
            escape_latex_text(segaux_cell),
        ]
        if has_c3c5:
            row_parts.extend([escape_latex_text(c3_cell), escape_latex_text(c5_cell)])
        if include_loss:
            row_parts.append(escape_latex_text(str(row["loss"])))
        row_parts.extend(metric_cells)
        result.append(" & ".join(row_parts) + r" \\")

    result += [r"\bottomrule", r"\end{tabular}", r"\vspace{0.15cm}"]
    return result


def _append_dataset_mean_table_lines(
    lines: List[str],
    rows: List[Dict[str, object]],
    dataset_name: str,
    include_std: bool = True,
    include_loss: bool = True,
    start_num: int = 1,
) -> None:
    metric_specs = dataset_metric_specs(dataset_name)
    rank_map = {
        key: _rank_row_indices(rows, key, higher_is_better=higher_is_better)
        for key, _, higher_is_better in metric_specs
    }
    lines += [r"\begin{table}[H]", r"\centering"]
    lines.extend(_build_tabular_lines(rows, dataset_name, include_std, include_loss, start_num, rank_map, 0))


def write_experiment_mean_latex(path: str, title: str, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a2paper, landscape, margin=1in]{geometry}",
        r"\usepackage{amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{float}",
        r"\usepackage{multicol}",
        r"\usepackage{caption}",
        r"\usepackage{multirow}",
        r"\captionsetup[table]{font=scriptsize}",
        r"\begin{document}",
        rf"\renewcommand{{\arraystretch}}{{{LATEX_TABLE_ARRAYSTRETCH}}}",
    ]

    rows_by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        dataset_name = str(row.get("dataset", "unknown_dataset"))
        if not _is_counted_dataset_name(dataset_name):
            continue
        rows_by_dataset.setdefault(dataset_name, []).append(row)

    has_previous_dataset = False
    if len(rows_by_dataset) == 1:
        dataset_name, dataset_rows = next(iter(rows_by_dataset.items()))
        has_previous_dataset = _append_page_break(lines, has_previous_dataset)
        _append_dataset_mean_table_lines(lines, dataset_rows, dataset_name)
        lines += [
            rf"\caption{{Cross-experiment mean summary ({escape_latex_text(dataset_name)})}}",
            r"\end{table}",
        ]
    else:
        grouped_rows: Dict[int, List[Tuple[str, List[Dict[str, object]]]]] = {}
        for dataset_name, dataset_rows in rows_by_dataset.items():
            group_order, _ = _dataset_summary_group(dataset_name)
            grouped_rows.setdefault(group_order, []).append((dataset_name, dataset_rows))

        for group_order in sorted(grouped_rows):
            group_items = sorted(grouped_rows[group_order], key=lambda item: _dataset_group_sort_key(item[0]))
            if not group_items:
                continue
            for dataset_name, dataset_rows in group_items:
                if len(dataset_rows) == 0:
                    continue
                has_previous_dataset = _append_page_break(lines, has_previous_dataset)
                _append_dataset_group_heading(lines, dataset_name)
                _append_dataset_mean_table_lines(lines, dataset_rows, dataset_name)
                lines += [
                    rf"\caption{{Cross-experiment mean summary ({escape_latex_text(dataset_name)})}}",
                    r"\end{table}",
                ]

    lines += [
        r"\end{document}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _row_path_info(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "experiment": str(row.get("experiment_source", row.get("experiment", ""))),
        "path": str(row.get("root", "")),
        "dice": float(row.get("best_dice", float("nan"))),
    }


def write_ablation_latex(
    path: str,
    rows: List[Dict[str, object]],
    include_buckets: Optional[Set[str]] = None,
    paths_out: Optional[Dict] = None,
) -> None:
    datasets = set()
    for row in rows:
        dataset_name = str(row.get("dataset", "unknown"))
        if not _is_counted_dataset_name(dataset_name):
            continue
        datasets.add(dataset_name)

    def _normalize_octa_bucket(dataset_name: str) -> Optional[str]:
        normalized = _normalize_dataset_name(dataset_name)
        if "octa" not in normalized:
            return None
        if "6m" in normalized:
            return "OCTA6M"
        if "3m" in normalized:
            return "OCTA3M"
        return None

    def _ablation_bucket_key(dataset_name: str) -> Tuple[int, str, str]:
        normalized = _normalize_dataset_name(dataset_name)
        if normalized == "drive":
            return 0, "DRIVE", natural_sort_key(dataset_name)
        if normalized == "deepcrack":
            return 1, "DeepCrack", natural_sort_key(dataset_name)
        if normalized in {"chase", "chasedb1"}:
            return 2, "CHASE", natural_sort_key(dataset_name)
        octa_bucket = _normalize_octa_bucket(dataset_name)
        if octa_bucket == "OCTA6M":
            return 3, octa_bucket, natural_sort_key(dataset_name)
        if octa_bucket == "OCTA3M":
            return 4, octa_bucket, natural_sort_key(dataset_name)
        return 5, "Others", natural_sort_key(dataset_name)

    dataset_buckets_map: Dict[str, List[str]] = {
        "DRIVE": [],
        "DeepCrack": [],
        "CHASE": [],
        "OCTA6M": [],
        "OCTA3M": [],
        "Others": [],
    }
    for dataset_name in sorted(datasets, key=_ablation_bucket_key):
        _, bucket_name, _ = _ablation_bucket_key(dataset_name)
        dataset_buckets_map[bucket_name].append(dataset_name)

    if include_buckets is not None:
        dataset_buckets_map = {
            bucket_name: bucket_rows
            for bucket_name, bucket_rows in dataset_buckets_map.items()
            if bucket_name in include_buckets
        }

    dataset_buckets: List[Tuple[str, List[str]]] = [
        (bucket_name, bucket_rows)
        for bucket_name, bucket_rows in dataset_buckets_map.items()
        if bucket_rows
    ]

    def row_label_mode(row: Dict[str, object]) -> str:
        label_mode = str(row.get("label_mode", ""))
        if label_mode in ALLOWED_LABEL_MODES:
            return label_mode
        return infer_label_mode_from_experiment(row.get("experiment", "unknown"))

    def fusion_objective_spec(row: Dict[str, object]) -> Tuple[str, str]:
        conn_fusion = str(row.get("conn_fusion", "none"))
        fusion_branches = row.get("fusion_branches", "none")
        residual_scale = row.get("fusion_residual_scale")
        decoder_fusion = row.get("decoder_fusion", "none")
        return (
            _format_fusion_display_name(conn_fusion, fusion_branches),
            _latex_ablation_fusion_spec(
                conn_fusion,
                residual_scale,
                decoder_fusion,
            ),
        )

    def decoder_option_spec(row: Dict[str, object]) -> Tuple[str]:
        _fusion_label, dec_label = _table_fusion_and_decoder_labels(
            row.get("conn_fusion", "none"),
            row.get("fusion_residual_scale"),
            row.get("fusion_branches", "none"),
            row.get("decoder_fusion", "none"),
            row.get("seg_aux_weight"),
            row.get("seg_aux_variant", "none"),
        )
        return (str(dec_label),)

    def run_rank_key(row: Dict[str, object]) -> Tuple[object, ...]:
        dice = float(row.get("best_dice", float("nan")))
        iou = float(row.get("best_jac", float("nan")))
        dice_key = dice if not math.isnan(dice) else float("-inf")
        iou_key = iou if not math.isnan(iou) else float("-inf")
        return (
            dice_key,
            iou_key,
            tuple(-x if isinstance(x, int) else x for x in experiment_sort_key(row)),
            natural_sort_key(str(row.get("experiment", "unknown"))),
        )

    def _fmt_signed_delta(value: float) -> str:
        if math.isnan(value):
            return "-"
        return f"{value:+.4f}"

    def _fmt_latex_delta_colored(value: float) -> str:
        return _fmt_latex_delta_colored_with_direction(value, higher_is_better=True)

    def _fmt_latex_delta_colored_with_direction(value: float, higher_is_better: bool) -> str:
        if math.isnan(value):
            return "-"
        signed_value = f"{value:+.4f}"
        if value == 0:
            return signed_value
        is_improvement = value > 0 if higher_is_better else value < 0
        if is_improvement:
            return rf"\textcolor{{green!60!black}}{{{signed_value}}}"
        return rf"\textcolor{{red!70!black}}{{{signed_value}}}"

    def _fmt_latex_value_with_delta(
        value: float,
        delta: float,
        higher_is_better: bool,
    ) -> str:
        if math.isnan(value):
            return r"\shortstack{- \\ (-)}"
        value_text = _fmt_latex_ranked_value(value, False, False)
        delta_text = _fmt_latex_delta_colored_with_direction(delta, higher_is_better)
        return rf"\shortstack{{{value_text} \\ ({delta_text})}}"

    # Label Mode / Loss / Fusion+Decoder/SegAux sweep tables (formerly Tables
    # 7-12) are no longer emitted in the ablation PDF; only Tables 1-6
    # (category-grouped, S-ablation, branch-letter ablation) are kept.
    variable_groups: List[Dict[str, object]] = []

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a2paper, landscape, margin=1in]{geometry}",
        r"\usepackage{amssymb}",
        r"\newcommand{\cmark}{\checkmark}",
        r"\newcommand{\xmark}{\ensuremath{\times}}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{float}",
        r"\usepackage{multicol}",
        r"\usepackage{caption}",
        r"\usepackage{multirow}",
        r"\captionsetup[table]{font=scriptsize}",
        r"\begin{document}",
        rf"\renewcommand{{\arraystretch}}{{{LATEX_TABLE_ARRAYSTRETCH}}}",
        r"\section*{Ablation Studies (Category-grouped Best Runs)}",
    ]

    def get_ranks(values: List[float], higher_is_better: bool = True) -> Tuple[Set[int], Set[int]]:
        valid_vals = [v for v in values if not math.isnan(v)]
        if not valid_vals:
            return set(), set()
        ranked = sorted(set(valid_vals), reverse=higher_is_better)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        best_idx = {i for i, v in enumerate(values) if v == best}
        second_idx = {i for i, v in enumerate(values) if v == second} if second is not None else set()
        return best_idx, second_idx

    def _delta_effect(delta: float, higher_is_better: bool) -> float:
        if math.isnan(delta):
            return float("nan")
        return delta if higher_is_better else -delta

    def _delta_cell_with_gradient(
        delta: float,
        higher_is_better: bool,
        metric_min_effect: float,
        metric_max_effect: float,
    ) -> str:
        if math.isnan(delta):
            return "-"
        delta_txt = rf"({_fmt_latex_delta_colored_with_direction(delta, higher_is_better)})"
        effect = _delta_effect(delta, higher_is_better)
        if math.isnan(effect):
            return delta_txt
        if effect > 0 and metric_max_effect > 0:
            strength = max(0.0, min(1.0, effect / metric_max_effect))
            pct = int(round(6 + 24 * strength))
            return rf"\cellcolor{{green!{pct}!white}}{delta_txt}"
        if effect < 0 and metric_min_effect < 0:
            strength = max(0.0, min(1.0, abs(effect) / abs(metric_min_effect)))
            pct = int(round(6 + 24 * strength))
            return rf"\cellcolor{{red!{pct}!white}}{delta_txt}"
        return delta_txt

    # Additional table: from each dataset best run, switch one element at a time
    # and report the best available alternative with +/- deltas.
    drop_axes = [
        ("label_mode", "Label Mode", lambda r: row_label_mode(r)),
        ("dec", "SegAux", lambda r: decoder_option_spec(r)[0]),
        ("fusion", "Fusion", lambda r: str(fusion_objective_spec(r))),
    ]
    preferred_drop_one_losses = ("gjml_sf_l1", "smooth_l1")
    preferred_drop_one_loss_set = set(preferred_drop_one_losses)

    def _is_proposed_row(row: Dict[str, object]) -> bool:
        try:
            seg_aux_weight = float(row.get("seg_aux_weight"))
            c3 = float(row.get("conn_aux_c3_weight"))
            c5 = float(row.get("conn_aux_c5_weight"))
        except (TypeError, ValueError):
            return False
        return (
            row_label_mode(row) == "dist_inverted"
            and row.get("conn_num", "unknown") == 8
            and str(row.get("conn_layout", "default")) == "default"
            and str(row.get("conn_fusion", "none")) == "dg_dsf"
            and str(row.get("fusion_branches", "none")) in ("none", "io")
            and str(row.get("loss", "unknown")) == "smooth_l1"
            and math.isclose(seg_aux_weight, 0.5)
            and math.isclose(c3, 0.5)
            and math.isclose(c5, 0.3)
        )

    drop_metric_specs = [
        ("best_dice", r"Dice $\uparrow$", True),
        ("best_jac", r"IoU $\uparrow$", True),
        ("best_cldice", r"clDice $\uparrow$", True),
        ("best_betti_error_0", r"Err $(\beta_0)$ $\downarrow$", False),
        ("best_betti_error_1", r"Err $(\beta_1)$ $\downarrow$", False),
        ("best_hd95", r"HD95 $\downarrow$", False),
    ]

    def _loss_toggle_target(loss_name: object) -> Optional[str]:
        loss = str(loss_name if loss_name is not None else "unknown")
        if loss == "gjml_sf_l1":
            return "smooth_l1"
        if loss == "smooth_l1":
            return "gjml_sf_l1"
        return None

    def _same_conn_config(a: Dict[str, object], b: Dict[str, object]) -> bool:
        return (
            a.get("conn_num", "unknown") == b.get("conn_num", "unknown")
            and str(a.get("conn_layout", "default")) == str(b.get("conn_layout", "default"))
        )

    def _is_plain_binary8_baseline(row: Dict[str, object]) -> bool:
        return (
            row_label_mode(row) == "binary"
            and row.get("conn_num", "unknown") == 8
            and str(row.get("conn_layout", "default")) == "default"
            and str(row.get("conn_fusion", "none")) == "none"
            and decoder_option_spec(row)[0] == "none"
            and str(row.get("loss", "unknown")) == "bce"
        )

    def _row_has_madf(row: Dict[str, object]) -> bool:
        return str(row.get("conn_fusion", "none")) != "none"

    def _row_has_segaux(row: Dict[str, object]) -> bool:
        try:
            w = float(row.get("seg_aux_weight") or 0)
            return w > 0
        except (TypeError, ValueError):
            return False

    def _normalize_drop_dec_label(dec_label: object) -> str:
        label = str(dec_label if dec_label is not None else "none")
        m = re.fullmatch(r"(w[0-9]*\.?[0-9]+)\([^)]+\)", label)
        if m is not None:
            return m.group(1)
        if re.fullmatch(r"segaux\([^)]+\)", label) is not None:
            return "segaux"
        return label

    def _same_fusion_profile(a: Dict[str, object], b: Dict[str, object]) -> bool:
        return (
            normalize_conn_fusion_name(a.get("conn_fusion", "none"))
            == normalize_conn_fusion_name(b.get("conn_fusion", "none"))
        )

    def _drop_one_variant_display_name(name: str) -> str:
        mapping = {
            "BEST": "Proposed",
            "-Label Mode": r"LATEX:Dist. $\rightarrow$ Binary",
            "-SegAux": "-SAL",
            "-Fusion": "-MDF",
            "BASE": "baseline []",
        }
        return mapping.get(name, name)

    preferred_dist_to_binary_sources = {
        "binary_dg_direct_8_bce_segaux_w0.5",
    }

    for bucket_name, bucket_datasets in dataset_buckets:
        table_dataset_groups: List[Tuple[str, List[str]]] = []
        if bucket_name == "CREMI, DRIVE":
            if "cremi" in bucket_datasets:
                table_dataset_groups.append(("CREMI", ["cremi"]))
            if "drive" in bucket_datasets:
                table_dataset_groups.append(("DRIVE", ["drive"]))
        else:
            for ds in bucket_datasets:
                table_dataset_groups.append((ds, [ds]))

        for table_label, table_datasets in table_dataset_groups:
            all_ds_rows: List[Dict[str, object]] = []
            for ds in table_datasets:
                ds_rows = [r for r in rows if str(r.get("dataset", "unknown")) == ds]
                if _normalize_dataset_name(ds) == "drive":
                    ds_rows = [
                        r for r in ds_rows
                        if str(r.get("loss", "unknown")) == "smooth_l1"
                        or row_label_mode(r) == "binary"
                    ]
                all_ds_rows.extend(ds_rows)

            if not all_ds_rows:
                continue

            # Find global best (proposed config, or fallback)
            proposed_rows = [r for r in all_ds_rows if _is_proposed_row(r)]
            if proposed_rows:
                global_best_row = max(proposed_rows, key=run_rank_key)
            else:
                preferred_rows = [
                    r for r in all_ds_rows
                    if str(r.get("loss", "unknown")) in preferred_drop_one_loss_set
                ]
                global_best_row = max(preferred_rows or all_ds_rows, key=run_rank_key)

            if paths_out is not None:
                paths_out.setdefault(table_label, {}).setdefault("ablation_table", {})["proposed"] = _row_path_info(global_best_row)

            # Label-mode groups. The proposed label mode keeps the full MADF×SegAux
            # grid; the other label modes only keep the full (MADF, SegAux) config.
            full_combos = [(True, True), (True, False), (False, True), (False, False)]
            lm_groups = [
                ("dist_inverted", ["Distance Map", "(Inverted)"], full_combos),
                ("dist", ["Distance Map"], [(True, True)]),
                ("binary", ["Binary"], [(True, True)]),
            ]

            # For each (lm, has_madf, has_segaux): find best matching row.
            # MADF candidates are restricted to the same fusion type (and c3/c5 weights) as
            # the global best (dg_dsf) so the ablation specifically measures the effect of
            # that fusion, not any fusion or a re-tuned c3/c5 sweep. SegAux candidates are
            # likewise restricted to the same seg_aux_weight as the global best, so the
            # ablation measures the effect of the flag, not a re-tuned weight. final_decision
            # is pinned across every combo so only the one ablated factor (label mode /
            # MADF / SegAux) varies -- otherwise a combo cell can silently pick a
            # differently-configured run (e.g. a final_decision=weighted_mean sweep variant)
            # that never gets compared against the actual proposed anchor.
            global_best_fusion = str(global_best_row.get("conn_fusion", "dg_dsf"))
            global_best_fusion_branches = str(global_best_row.get("fusion_branches", "none"))
            global_best_final_decision = str(global_best_row.get("final_decision", "max"))
            global_best_c3 = global_best_row.get("conn_aux_c3_weight")
            global_best_c5 = global_best_row.get("conn_aux_c5_weight")
            try:
                global_best_segaux_weight = float(global_best_row.get("seg_aux_weight"))
            except (TypeError, ValueError):
                global_best_segaux_weight = None

            def _matches_global_weight(value: object, target: object) -> bool:
                if target is None:
                    return True
                if value is None:
                    return False
                try:
                    return math.isclose(float(value), float(target), abs_tol=1e-6)
                except (TypeError, ValueError):
                    return False

            table_data: Dict[Tuple[str, bool, bool], Tuple[Optional[Dict[str, object]], List[float], List[float]]] = {}
            for lm_key, _lm_labels, lm_combos in lm_groups:
                for has_madf, has_segaux in lm_combos:
                    def _candidates(match_madf_shape: bool) -> List[Dict[str, object]]:
                        return [
                            r for r in all_ds_rows
                            if row_label_mode(r) == lm_key
                            and _row_has_madf(r) == has_madf
                            and _row_has_segaux(r) == has_segaux
                            and _same_conn_config(r, global_best_row)
                            and str(r.get("final_decision", "max")) == global_best_final_decision
                            and (not has_madf or str(r.get("conn_fusion", "none")) == global_best_fusion)
                            and (
                                not has_madf
                                or not match_madf_shape
                                or (
                                    str(r.get("fusion_branches", "none")) == global_best_fusion_branches
                                    and _matches_global_weight(r.get("conn_aux_c3_weight"), global_best_c3)
                                    and _matches_global_weight(r.get("conn_aux_c5_weight"), global_best_c5)
                                )
                            )
                            and (
                                not has_segaux
                                or global_best_segaux_weight is None
                                or math.isclose(float(r.get("seg_aux_weight") or 0), global_best_segaux_weight)
                            )
                        ]

                    # Within the anchor's own label mode, pin the branch composition and
                    # c3/c5 weights so the ablated factor is the only thing that varies.
                    # Across label modes, don't: the Dist. -> Binary cell is required to
                    # come from a purpose-built binary baseline
                    # (binary_dg_direct_A_8_bce_segaux_w0.5), which was trained separately
                    # and carries its own branch composition. Pinning there would drop it
                    # in favour of a drop-one derived row, or blank the cell outright --
                    # the same silent hole that once blanked the DRIVE Proposed row in
                    # summary_comparison_models. Fall back to the looser match either way
                    # rather than emitting NaN.
                    match_madf_shape = lm_key == row_label_mode(global_best_row)
                    candidates = _candidates(match_madf_shape) or _candidates(False)
                    if not candidates:
                        table_data[(lm_key, has_madf, has_segaux)] = (
                            None,
                            [float("nan")] * len(drop_metric_specs),
                            [float("nan")] * len(drop_metric_specs),
                        )
                    else:
                        best = max(candidates, key=run_rank_key)
                        if paths_out is not None:
                            slot_key = f"{lm_key}+madf={has_madf}+segaux={has_segaux}"
                            paths_out[table_label]["ablation_table"][slot_key] = _row_path_info(best)
                        vals = [
                            float(best.get(mk, float("nan")))
                            for mk, _l, _h in drop_metric_specs
                        ]
                        stds = [
                            float(best.get(f"{mk}_std", float("nan")))
                            for mk, _l, _h in drop_metric_specs
                        ]
                        table_data[(lm_key, has_madf, has_segaux)] = (best, vals, stds)

            if all(entry[0] is None for entry in table_data.values()):
                continue

            # Compute rankings across all value rows (flat order: lm × combo)
            combo_order = [
                (lm_key, has_madf, has_segaux)
                for lm_key, _lm_labels, lm_combos in lm_groups
                for has_madf, has_segaux in lm_combos
            ]
            combo_rank_idx = {key: idx for idx, key in enumerate(combo_order)}
            all_value_rows_flat = [table_data[k][1] for k in combo_order]

            value_rank_maps_2x2: List[Tuple[Set[int], Set[int]]] = []
            for metric_idx, (_mk, _ml, higher_is_better) in enumerate(drop_metric_specs):
                metric_vals = [vals[metric_idx] for vals in all_value_rows_flat]
                value_rank_maps_2x2.append(get_ranks(metric_vals, higher_is_better))

            def _fmt_2x2_mean_std(
                value: float,
                std: float,
                combo_key: Tuple[str, bool, bool],
                metric_idx: int,
            ) -> Tuple[str, str]:
                rank_idx = combo_rank_idx[combo_key]
                best_set, second_set = value_rank_maps_2x2[metric_idx]
                return _fmt_latex_ranked_mean_std_marginal(
                    value, std, rank_idx in best_set, rank_idx in second_set
                )

            header_parts = ["Label", "MADF", "SegAux"]
            for _mk, ml, _h in drop_metric_specs:
                header_parts.extend([ml, r"\textcolor{gray}{(std)}"])
            lines.extend([
                r"\begin{table}[!t]",
                r"\centering",
                r"\tiny",
                rf"\begin{{tabular}}{{ccc|{'cc' * len(drop_metric_specs)}}}",
                r"\toprule",
                " & ".join(header_parts) + r" \\",
                r"\midrule",
            ])

            for group_idx, (lm_key, lm_label_lines, lm_combos) in enumerate(lm_groups):
                if group_idx > 0:
                    lines.append(r"\midrule")

                label_text = lm_label_lines[0]
                if len(lm_label_lines) > 1:
                    label_text = r"\shortstack{" + r" \\ ".join(lm_label_lines) + "}"
                if len(lm_combos) > 1:
                    label_text = rf"\multirow{{{len(lm_combos)}}}{{*}}{{{label_text}}}"

                for combo_idx, (has_madf, has_segaux) in enumerate(lm_combos):
                    combo_key = (lm_key, has_madf, has_segaux)
                    _best_row_entry, vals, stds = table_data[combo_key]
                    has_data = not all(math.isnan(v) for v in vals)

                    madf_mark = r"\cmark" if has_madf else r"\xmark"
                    segaux_mark = r"\cmark" if has_segaux else r"\xmark"

                    metric_cells: List[str] = []
                    for mi, (val, std) in enumerate(zip(vals, stds)):
                        if not has_data:
                            metric_cells.extend(["", ""])
                            continue
                        mean_txt, std_txt = _fmt_2x2_mean_std(val, std, combo_key, mi)
                        metric_cells.extend([mean_txt, std_txt])

                    label_cell = label_text if combo_idx == 0 else ""
                    lines.append(
                        label_cell + " & " + madf_mark + " & " + segaux_mark
                        + " & " + " & ".join(metric_cells) + r" \\"
                    )

            dataset_label_clean = table_label.replace("_", r"\_")
            dataset_label_id = table_label.lower().replace(" ", "_")
            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"\vspace{0.3cm}",
                rf"\caption{{Ablation study on {dataset_label_clean} dataset. "
                r"\textbf{Bold}: best, \underline{underline}: second best.}"
                rf"\label{{tab:ablation_{dataset_label_id}}}",
                r"\end{table}",
                "",
            ])

    # Emit all tables for a bucket first so table numbering becomes:
    # core bucket(1,2,3,...) then other bucket(...).
    def _get_s_variation_label(row: Dict[str, object]) -> Optional[str]:
        try:
            val = row.get("seg_aux_weight")
            seg_aux_weight = float(val) if val is not None else None
        except (TypeError, ValueError):
            seg_aux_weight = None

        label_mode = row_label_mode(row)
        conn_num = row.get("conn_num", "unknown")
        conn_layout = str(row.get("conn_layout", "default"))
        conn_fusion = str(row.get("conn_fusion", "none"))
        fusion_branches = str(row.get("fusion_branches", "none"))
        loss = str(row.get("loss", "unknown"))

        if label_mode != "dist_inverted" or conn_num != 8 or loss != "smooth_l1":
            return None

        # 1 baseline: plain dist_inverted + segaux w0.5 (no fusion, so no braces)
        if (
            conn_layout == "default"
            and conn_fusion == "none"
            and seg_aux_weight is not None and math.isclose(seg_aux_weight, 0.5)
        ):
            return "1"

        # 2 baseline: out8 plain (no fusion, so no braces)
        if (
            conn_layout == "out8"
            and conn_fusion == "none"
            and seg_aux_weight is None
        ):
            return "2"

        # DSF: connectivity-based fusion with io branches + w0.5
        if conn_fusion == "dsf" and seg_aux_weight is not None and math.isclose(seg_aux_weight, 0.5):
            if fusion_branches in ("none", "io"):
                return "DSF"
            return None

        # DG-DSF S ablation: proposed conditions (segaux=0.5, c3≈0.5)
        if conn_fusion == "dg_dsf" and seg_aux_weight is not None and math.isclose(seg_aux_weight, 0.5):
            c3w = row.get("conn_aux_c3_weight")
            proposed_c3 = c3w is not None and math.isclose(float(c3w), 0.5, abs_tol=1e-3)
            if not proposed_c3:
                # Non-proposed c3 weight: plain DG-DSF label (io only)
                if fusion_branches in ("none", "io"):
                    return "DG-DSF"
                return None
            # Proposed c3≈0.5 → S-set labels (DG-DSF is the proposed architecture,
            # so its branch sweep owns the bare {..} notation; the legacy dg_direct
            # sweep below is superseded and no longer contributes rows here).
            if fusion_branches in ("none", "io"):
                return r"\textbf{\{1,2\}}"
            if fusion_branches == "i":
                return "1"
            if fusion_branches == "ip":
                return r"\{1,3\}"
            if fusion_branches == "if":
                return r"\{1,4\}"
            if fusion_branches in ("iop", "imp"):
                return r"\{1,2,3\}"
            if fusion_branches in ("iopf", "impf"):
                return r"\{1,2,3,4\}"
            return None

        # Legacy dg_direct architecture: superseded by DG-DSF as the proposed
        # method, so its S-sweep no longer contributes rows to this table.
        return None

    s_var_order = [
        "1", "2",
        r"\textbf{\{1,2\}}", r"\{1,2,3\}", r"\{1,2,3,4\}",
    ]

    # i -> A, o -> B, m -> C branch-letter ablation (dg_dsf, proposed segaux/c3
    # conditions). Compares the proposed io={A,B} row against the middle-branch
    # variants im={A,C} and imo={A,B,C}, plus a plain conn24 (no fusion)
    # baseline labeled D.
    def _get_imo_label(row: Dict[str, object]) -> Optional[str]:
        try:
            val = row.get("seg_aux_weight")
            seg_aux_weight = float(val) if val is not None else None
        except (TypeError, ValueError):
            seg_aux_weight = None

        label_mode = row_label_mode(row)
        conn_num = row.get("conn_num", "unknown")
        conn_layout = str(row.get("conn_layout", "default"))
        loss = str(row.get("loss", "unknown"))
        conn_fusion = str(row.get("conn_fusion", "none"))
        fusion_branches = str(row.get("fusion_branches", "none"))

        if label_mode != "dist_inverted" or loss != "smooth_l1":
            return None
        if seg_aux_weight is None or not math.isclose(seg_aux_weight, 0.5):
            return None

        # D: plain conn24 baseline (no fusion), for comparison against the
        # conn8 dg_dsf branch-letter variants below.
        if conn_num == 24:
            if conn_layout == "default" and conn_fusion == "none":
                return "D"
            return None

        if conn_num != 8 or conn_fusion != "dg_dsf":
            return None

        c3w = row.get("conn_aux_c3_weight")
        if c3w is None or not math.isclose(float(c3w), 0.5, abs_tol=1e-3):
            return None

        if fusion_branches == "i":
            return "A"
        if fusion_branches in ("none", "io"):
            return r"\textbf{\{A,B\}}"
        if fusion_branches == "im":
            return r"\{A,C\}"
        if fusion_branches == "imo":
            return r"\{A,B,C\}"
        return None

    imo_var_order = [
        "A", r"\textbf{\{A,B\}}", r"\{A,C\}", r"\{A,B,C\}", "D",
    ]

    for bucket_name, bucket_datasets in dataset_buckets:
        for ds in bucket_datasets:
            ds_rows = [r for r in rows if str(r.get("dataset", "unknown")) == ds]

            s_rows_dict = {}
            for r in ds_rows:
                lbl = _get_s_variation_label(r)
                if lbl is not None:
                    if lbl not in s_rows_dict or run_rank_key(r) > run_rank_key(s_rows_dict[lbl]):
                        s_rows_dict[lbl] = r

            if not s_rows_dict:
                continue

            if paths_out is not None:
                s_var_paths = {lbl: _row_path_info(s_rows_dict[lbl]) for lbl in s_var_order if lbl in s_rows_dict}
                paths_out.setdefault(ds, {})["s_variation"] = s_var_paths

            s_metric_specs = [
                ("best_dice", r"Dice $\uparrow$", True),
                ("best_jac", r"IoU $\uparrow$", True),
                ("best_cldice", r"clDice $\uparrow$", True),
                ("best_betti_error_0", r"Err $(\beta_0)$ $\downarrow$", False),
                ("best_betti_error_1", r"Err $(\beta_1)$ $\downarrow$", False),
                ("best_hd95", r"HD95 $\downarrow$", False),
            ]

            lines.extend([
                r"\begin{table}[H]",
                r"\centering",
                r"\tiny",
            ])

            # Header parts with standard deviation (similar to comparison models)
            header_parts = [r"$\mathcal{S}$"]
            for _, lbl, _ in s_metric_specs:
                header_parts.extend([lbl, r"\textcolor{gray}{(std)}"])

            table_columns = 1 + 2 * len(s_metric_specs)
            lines.append(rf"\begin{{tabular}}{{{'c' * table_columns}}}")
            lines.append(r"\toprule")
            lines.append(" & ".join(header_parts) + r" \\")
            lines.append(r"\midrule")

            metric_vals = {key: [] for key, _, _ in s_metric_specs}
            for lbl in s_var_order:
                if lbl in s_rows_dict:
                    row = s_rows_dict[lbl]
                    for key, _, _ in s_metric_specs:
                        metric_vals[key].append(float(row.get(key, float("nan"))))

            ranks = {}
            for key, _, hib in s_metric_specs:
                ranks[key] = get_ranks(metric_vals[key], hib)

            val_idx = 0
            for lbl in s_var_order:
                if lbl in s_rows_dict:
                    row = s_rows_dict[lbl]
                    row_cells = [lbl]
                    for key, _, _ in s_metric_specs:
                        val = float(row.get(key, float("nan")))
                        std = float(row.get(f"{key}_std", float("nan")))
                        best_idx, second_idx = ranks[key]

                        mean_txt, std_txt = _fmt_latex_ranked_mean_std_marginal(
                            val, std, val_idx in best_idx, val_idx in second_idx
                        )
                        row_cells.extend([mean_txt, std_txt])
                    lines.append(" & ".join(row_cells) + r" \\")
                    val_idx += 1

            ds_display = {"drive": "Drive", "deepcrack": "DeepCrack", "chase": "Chase", "isic": "ISIC"}.get(ds, ds)
            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"\vspace{0.15cm}",
                rf"\caption{{Ablation study on $\mathcal{{S}}$ ({escape_latex_text(ds_display)} dataset). \textbf{{Bold}}: best, \underline{{underline}}: second best.}}\label{{tab:ablation_s_{ds}}}",
                r"\end{table}",
                "",
            ])

    for bucket_name, bucket_datasets in dataset_buckets:
        for ds in bucket_datasets:
            ds_rows = [r for r in rows if str(r.get("dataset", "unknown")) == ds]

            imo_rows_dict = {}
            for r in ds_rows:
                lbl = _get_imo_label(r)
                if lbl is not None:
                    if lbl not in imo_rows_dict or run_rank_key(r) > run_rank_key(imo_rows_dict[lbl]):
                        imo_rows_dict[lbl] = r

            if not imo_rows_dict:
                continue

            if paths_out is not None:
                imo_paths = {lbl: _row_path_info(imo_rows_dict[lbl]) for lbl in imo_var_order if lbl in imo_rows_dict}
                paths_out.setdefault(ds, {})["imo_variation"] = imo_paths

            imo_metric_specs = [
                ("best_dice", r"Dice $\uparrow$", True),
                ("best_jac", r"IoU $\uparrow$", True),
                ("best_cldice", r"clDice $\uparrow$", True),
                ("best_betti_error_0", r"Err $(\beta_0)$ $\downarrow$", False),
                ("best_betti_error_1", r"Err $(\beta_1)$ $\downarrow$", False),
                ("best_hd95", r"HD95 $\downarrow$", False),
            ]

            lines.extend([
                r"\begin{table}[H]",
                r"\centering",
                r"\tiny",
            ])

            header_parts = [r"$\mathcal{S}$"]
            for _, lbl, _ in imo_metric_specs:
                header_parts.extend([lbl, r"\textcolor{gray}{(std)}"])

            table_columns = 1 + 2 * len(imo_metric_specs)
            lines.append(rf"\begin{{tabular}}{{{'c' * table_columns}}}")
            lines.append(r"\toprule")
            lines.append(" & ".join(header_parts) + r" \\")
            lines.append(r"\midrule")

            metric_vals = {key: [] for key, _, _ in imo_metric_specs}
            for lbl in imo_var_order:
                if lbl in imo_rows_dict:
                    row = imo_rows_dict[lbl]
                    for key, _, _ in imo_metric_specs:
                        metric_vals[key].append(float(row.get(key, float("nan"))))

            ranks = {}
            for key, _, hib in imo_metric_specs:
                ranks[key] = get_ranks(metric_vals[key], hib)

            val_idx = 0
            for lbl in imo_var_order:
                if lbl in imo_rows_dict:
                    row = imo_rows_dict[lbl]
                    row_cells = [lbl]
                    for key, _, _ in imo_metric_specs:
                        val = float(row.get(key, float("nan")))
                        std = float(row.get(f"{key}_std", float("nan")))
                        best_idx, second_idx = ranks[key]

                        mean_txt, std_txt = _fmt_latex_ranked_mean_std_marginal(
                            val, std, val_idx in best_idx, val_idx in second_idx
                        )
                        row_cells.extend([mean_txt, std_txt])
                    lines.append(" & ".join(row_cells) + r" \\")
                    val_idx += 1

            ds_display = {"drive": "Drive", "deepcrack": "DeepCrack", "chase": "Chase", "isic": "ISIC"}.get(ds, ds)
            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"\vspace{0.15cm}",
                rf"\caption{{Ablation study on branch composition, A=inner(i), B=outer(o), C=middle(m), "
                rf"D=conn24 (no fusion) "
                rf"({escape_latex_text(ds_display)} dataset). \textbf{{Bold}}: best, \underline{{underline}}: second best.}}\label{{tab:ablation_imo_{ds}}}",
                r"\end{table}",
                "",
            ])

    for v_group in variable_groups:
        for bucket_name, bucket_datasets in dataset_buckets:
            var_name = v_group["name"]
            headers = v_group["headers"]
            var_func = v_group["func"]
            row_filter = v_group.get("filter", lambda r: True)

            val_set = set()
            for row in rows:
                if row_filter(row):
                    val_set.add(var_func(row))

            def tuple_sort_key(tup):
                custom_sort_key = v_group.get("sort_key")
                if custom_sort_key is not None:
                    return custom_sort_key(tup)
                return [natural_sort_key(str(x)) for x in tup]

            val_list = sorted(list(val_set), key=tuple_sort_key)
            selected_rows: Dict[str, Dict[Tuple[object, ...], Dict[str, object]]] = {
                ds: {} for ds in bucket_datasets
            }

            for row in rows:
                if not row_filter(row):
                    continue
                ds = str(row.get("dataset", "unknown"))
                if ds not in selected_rows:
                    continue

                # User request: Only use smooth_l1 for DRIVE dataset ablation comparisons
                # (except when the ablation variable itself is Loss)
                if _normalize_dataset_name(ds) == "drive" and var_name != "Loss":
                    if str(row.get("loss", "unknown")) != "smooth_l1":
                        continue

                val = var_func(row)
                existing = selected_rows[ds].get(val)
                if existing is None or run_rank_key(row) > run_rank_key(existing):
                    selected_rows[ds][val] = row

            if paths_out is not None:
                pass  # variable_groups excluded from paths JSON (Tables 5+ not in ablation_paths)

            ablation_metric_specs = [
                ("best_dice", r"Dice $\uparrow$", True),
                ("best_jac", r"IoU $\uparrow$", True),
                ("best_cldice", r"clDice $\uparrow$", True),
                ("best_betti_error_0", r"Err $(\beta_0)$ $\downarrow$", False),
                ("best_betti_error_1", r"Err $(\beta_1)$ $\downarrow$", False),
                ("best_hd95", r"HD95 $\downarrow$", False),
            ]
            n_metrics = len(ablation_metric_specs)

            dataset_metrics = {
                ds: {key: [] for key, _label, _hib in ablation_metric_specs}
                for ds in bucket_datasets
            }

            for ds in bucket_datasets:
                for v in val_list:
                    selected = selected_rows[ds].get(v)
                    for key, _label, _hib in ablation_metric_specs:
                        if selected is None:
                            dataset_metrics[ds][key].append(float("nan"))
                        else:
                            dataset_metrics[ds][key].append(float(selected.get(key, float("nan"))))

            dataset_ranks = {}
            for ds in bucket_datasets:
                dataset_ranks[ds] = {}
                for key, _label, higher_is_better in ablation_metric_specs:
                    best_idx, second_idx = get_ranks(dataset_metrics[ds][key], higher_is_better)
                    dataset_ranks[ds][key] = (best_idx, second_idx)

            lines.extend([
                r"\begin{table}[H]",
                r"\centering",
                r"\tiny",
            ])

            num_var_cols = len(headers) + 1
            col_str = "l" * num_var_cols + ("c" * n_metrics) * len(bucket_datasets)
            lines.append(rf"\begin{{tabular}}{{{col_str}}}")
            lines.append(r"\toprule")

            header1 = [r"\multirow{2}{*}{No.}"]
            for h in headers:
                if isinstance(h, str) and h.startswith("LATEX:"):
                    header_txt = h[len("LATEX:"):]
                else:
                    header_txt = escape_latex_text(str(h))
                header1.append(rf"\multirow{{2}}{{*}}{{{header_txt}}}")
            for ds in bucket_datasets:
                header1.append(rf"\multicolumn{{{n_metrics}}}{{c}}{{{escape_latex_text(ds)}}}")
            lines.append(" & ".join(header1) + r" \\")

            cmidrules = []
            current_col = num_var_cols + 1
            for _ in bucket_datasets:
                cmidrules.append(rf"\cmidrule(lr){{{current_col}-{current_col + n_metrics - 1}}}")
                current_col += n_metrics
            if cmidrules:
                lines.append(" ".join(cmidrules))

            header2 = [""] * num_var_cols
            for _ in bucket_datasets:
                header2.extend(label for _key, label, _hib in ablation_metric_specs)
            lines.append(" & ".join(header2) + r" \\")
            lines.append(r"\midrule")

            group_by_conn_fusion = var_name == "Fusion + Decoder/SegAux"
            prev_group_key: Optional[str] = None
            for idx, v in enumerate(val_list):
                current_group_key = str(v[0]) if group_by_conn_fusion and len(v) > 0 else None
                if group_by_conn_fusion and prev_group_key is not None and current_group_key != prev_group_key:
                    lines.append(r"\midrule")
                prev_group_key = current_group_key

                row_str_parts = [str(idx + 1)]
                for x in v:
                    x_str = str(x)
                    if x_str.startswith("LATEX:"):
                        row_str_parts.append(x_str[len("LATEX:"):])
                    else:
                        row_str_parts.append(escape_latex_text(x_str))
                for ds in bucket_datasets:
                    selected = selected_rows[ds].get(v)
                    for key, _label, _hib in ablation_metric_specs:
                        value = (
                            float("nan")
                            if selected is None
                            else float(selected.get(key, float("nan")))
                        )
                        best_idx, second_idx = dataset_ranks[ds][key]
                        row_str_parts.append(
                            _fmt_latex_ranked_value(value, idx in best_idx, idx in second_idx)
                        )
                lines.append(" & ".join(row_str_parts) + r" \\")

            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"\vspace{0.15cm}",
                rf"\caption{{Ablation on {escape_latex_text(var_name)} ({escape_latex_text(bucket_name)})}}",
                r"\end{table}",
                "",
            ])

    lines.append(r"\end{document}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _append_comparison_model_table_lines(
    lines: List[str],
    rows: List[Dict[str, object]],
    dataset_name: str,
) -> None:
    metric_specs = dataset_metric_specs(dataset_name)
    rank_map = {
        key: _rank_row_indices(rows, key, higher_is_better=higher_is_better)
        for key, _, higher_is_better in metric_specs
    }

    table_columns = 2 + 2 * len(metric_specs)
    header_parts = ["No.", "Model"]
    for _, label, _ in metric_specs:
        header_parts.extend([label, r"\textcolor{gray}{(std)}"])

    lines += [
        r"\begin{table}[H]",
        r"\centering",
        rf"\begin{{tabular}}{{{'l' + 'c' * (table_columns - 1)}}}",
        r"\toprule",
        " & ".join(header_parts) + r" \\",
        r"\midrule",
    ]

    for idx, row in enumerate(rows):
        model_name = str(row.get("model_name", "unknown"))
        metric_cells: List[str] = []
        for key, _, _ in metric_specs:
            best_indices, second_indices = rank_map[key]
            mean_txt, std_txt = _fmt_latex_ranked_mean_std_marginal(
                float(row.get(key, float("nan"))),
                float(row.get(f"{key}_std", float("nan"))),
                idx in best_indices,
                idx in second_indices,
            )
            metric_cells.extend([mean_txt, std_txt])
        row_parts = [f"{idx + 1}", escape_latex_text(model_name)]
        row_parts.extend(metric_cells)
        lines.append(" & ".join(row_parts) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]


def write_comparison_models_latex(
    path: str,
    rows_by_dataset: Dict[str, List[Dict[str, object]]],
) -> None:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a2paper, landscape, margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{float}",
        r"\usepackage[table]{xcolor}",
        r"\begin{document}",
        rf"\renewcommand{{\arraystretch}}{{{LATEX_TABLE_ARRAYSTRETCH}}}",
    ]

    grouped_rows: Dict[int, List[Tuple[str, List[Dict[str, object]]]]] = {}
    for dataset_name, rows in rows_by_dataset.items():
        if not _is_counted_dataset_name(dataset_name):
            continue
        group_order, _ = _dataset_summary_group(dataset_name)
        grouped_rows.setdefault(group_order, []).append((dataset_name, rows))

    has_previous_dataset = False
    for group_order in sorted(grouped_rows):
        group_items = sorted(grouped_rows[group_order], key=lambda item: _dataset_group_sort_key(item[0]))
        for dataset_name, rows in group_items:
            if not rows:
                continue
            sorted_rows = sorted(rows, key=lambda r: natural_sort_key(str(r.get("model_name", ""))))
            has_previous_dataset = _append_page_break(lines, has_previous_dataset)
            _append_dataset_group_heading(lines, dataset_name)
            _append_comparison_model_table_lines(lines, sorted_rows, dataset_name)
            lines += [
                rf"\caption{{Comparison models ({escape_latex_text(dataset_name)})}}",
                r"\end{table}",
            ]

    lines += [r"\end{document}"]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_experiment_mean_dataset_tables_latex(
    path: str,
    title: str,
    rows_by_dataset: Dict[str, List[Dict[str, object]]],
) -> None:
    lines = [
        r"\documentclass[9pt]{extarticle}",
        r"\usepackage[a2paper,landscape,margin=0.8in]{geometry}",
        r"\usepackage{amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{float}",
        r"\usepackage{multicol}",
        r"\usepackage{caption}",
        r"\usepackage{multirow}",
        r"\captionsetup[table]{font=scriptsize}",
        r"\begin{document}",
    ]

    grouped_rows: Dict[int, List[Tuple[str, List[Dict[str, object]]]]] = {}
    for dataset_name, rows in rows_by_dataset.items():
        if not _is_counted_dataset_name(dataset_name):
            continue
        group_order, _ = _dataset_summary_group(dataset_name)
        grouped_rows.setdefault(group_order, []).append((dataset_name, rows))

    has_previous_dataset = False
    for group_order in sorted(grouped_rows):
        group_items = sorted(grouped_rows[group_order], key=lambda item: _dataset_group_sort_key(item[0]))
        if not group_items:
            continue
        for dataset_name, rows in group_items:
            if len(rows) == 0:
                continue
            has_previous_dataset = _append_page_break(lines, has_previous_dataset)
            _append_dataset_group_heading(lines, dataset_name)
            rows_by_loss: Dict[str, List[Dict[str, object]]] = {}
            for row in rows:
                loss = str(row.get("loss", "unknown"))
                rows_by_loss.setdefault(loss, []).append(row)

            ordered_losses = sorted(rows_by_loss.keys(), key=_loss_sort_key)
            next_num = 1
            for loss in ordered_losses:
                loss_rows = rows_by_loss[loss]
                if len(loss_rows) == 0:
                    continue
                caption_line = rf"\caption{{Cross-experiment mean summary ({escape_latex_text(dataset_name)}, Loss={escape_latex_text(loss)})}}"
                if len(loss_rows) > 50:
                    # Two-column layout: rank across all rows, split into two halves.
                    # Pre-compute has_c3c5 from the full set so both halves share the
                    # same column count — avoiding mismatched side-by-side tables.
                    metric_specs = dataset_metric_specs(dataset_name)
                    rank_map = {
                        key: _rank_row_indices(loss_rows, key, higher_is_better=hib)
                        for key, _, hib in metric_specs
                    }
                    global_has_c3c5 = True
                    mid = (len(loss_rows) + 1) // 2
                    first_half = loss_rows[:mid]
                    second_half = loss_rows[mid:]
                    tab1 = _build_tabular_lines(first_half, dataset_name, False, False, next_num, rank_map, 0, force_has_c3c5=global_has_c3c5)
                    tab2 = _build_tabular_lines(second_half, dataset_name, False, False, next_num + mid, rank_map, mid, force_has_c3c5=global_has_c3c5)
                    lines += [
                        r"\begin{table}[H]",
                        r"\begin{minipage}[t]{0.49\linewidth}",
                        r"\centering",
                    ]
                    lines.extend(tab1)
                    lines += [
                        r"\end{minipage}",
                        r"\hfill",
                        r"\begin{minipage}[t]{0.49\linewidth}",
                        r"\centering",
                    ]
                    lines.extend(tab2)
                    lines += [r"\end{minipage}"]
                else:
                    _append_dataset_mean_table_lines(lines, loss_rows, dataset_name, include_std=False, include_loss=False, start_num=next_num)
                next_num += len(loss_rows)
                lines.extend([caption_line, r"\end{table}"])

            _append_external_baseline_table_lines(lines, dataset_name, start_num=1)

    lines += [
        r"\end{document}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for key, value in replacements.items():
        escaped = escaped.replace(key, value)

    # Use \times for patterns like 1x1
    escaped = re.sub(r'(\d)x(\d)', r'$\1\\times \2$', escaped)
    # Use en-dash for octa500-3M etc to satisfy chktex
    escaped = re.sub(r'([a-zA-Z0-9])-([a-zA-Z0-9])', r'\1--\2', escaped)

    return escaped


def build_pdf(tex_path: str, pdf_out_path: Optional[str] = None) -> str:
    out_dir = os.path.dirname(os.path.abspath(tex_path))
    tex_file = os.path.basename(tex_path)
    tex_stem, _ = os.path.splitext(tex_file)
    generated_pdf_path = os.path.join(out_dir, f"{tex_stem}.pdf")
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_file,
    ]
    env = os.environ.copy()
    output_tex_path = os.path.join(ROOT_DIR, "output")
    texinputs_prefix = output_tex_path + os.pathsep
    current_texinputs = env.get("TEXINPUTS", "")
    if texinputs_prefix not in current_texinputs.split(os.pathsep):
        env["TEXINPUTS"] = texinputs_prefix + current_texinputs
    subprocess.run(cmd, cwd=out_dir, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    target_pdf_path = (
        os.path.abspath(pdf_out_path)
        if pdf_out_path is not None
        else generated_pdf_path
    )
    if os.path.abspath(generated_pdf_path) != target_pdf_path:
        os.makedirs(os.path.dirname(target_pdf_path), exist_ok=True)
        shutil.move(generated_pdf_path, target_pdf_path)
    return target_pdf_path


def root_display_name(root: str, input_root: str) -> str:
    rel_path = os.path.relpath(root, input_root)
    if rel_path == ".":
        return os.path.basename(os.path.normpath(root))
    return rel_path


def root_output_label(root: str, input_root: str) -> str:
    return root_display_name(root, input_root).replace(os.sep, "_")


def move_non_pdf_files_to_dump(output_dir: str, dump_dir: str) -> None:
    for entry in os.listdir(output_dir):
        src_path = os.path.join(output_dir, entry)
        if not os.path.isfile(src_path):
            continue
        if entry.lower().endswith(".pdf") or entry.lower().endswith(".png"):
            continue

        dst_path = os.path.join(dump_dir, entry)
        if os.path.exists(dst_path):
            if os.path.isfile(dst_path):
                os.remove(dst_path)
            else:
                shutil.rmtree(dst_path)
        shutil.move(src_path, dst_path)


def normalize_aggregate_stem(output_stem: str) -> str:
    stem = str(output_stem).strip()
    if stem.endswith("_experiment_means"):
        return stem
    return f"{stem}_experiment_means"


def aggregate_output_stem(output_stem: str, include_all: bool) -> str:
    stem = normalize_aggregate_stem(output_stem)
    if include_all:
        return f"{stem}_all"
    return stem


def should_exclude_ambiguous_scaled_sum_row(exp_meta: Dict[str, object]) -> bool:
    """Drop legacy scaled_sum rows without explicit residual-scale spec.

    These runs were produced before `_rs<scale>` was appended to the
    experiment name, so the residual scale cannot be recovered reliably.
    """
    conn_fusion = str(exp_meta.get("conn_fusion", "none"))
    residual_scale = exp_meta.get("fusion_residual_scale")
    return (
        conn_fusion == "scaled_sum"
        and residual_scale is None
    )


def _root_has_valid_csv(root: str, folds: List[str], input_name: str) -> bool:
    """Return True if *any* fold in root has at least one parseable data row."""
    for fold in folds:
        try:
            csv_path = resolve_fold_csv_path(root, fold, input_name, allow_legacy_results=True)
        except FileNotFoundError:
            continue
        metrics = parse_fold_csv(csv_path)
        if metrics is not None:
            return True
    return False


def move_experiment_to_legacy(root: str, dry_run: bool = False) -> str:
    """Move an experiment folder to output/legacy/<dataset>/<name>/.

    Returns the destination path (whether or not the move was performed).
    If the destination already exists, appends a numeric suffix.
    """
    parent = os.path.dirname(root)
    dataset_dir_name = os.path.basename(parent)
    input_root_dir = os.path.dirname(parent)
    legacy_dir = os.path.join(input_root_dir, "legacy", dataset_dir_name)
    exp_name = os.path.basename(root)
    dest = os.path.join(legacy_dir, exp_name)
    if os.path.exists(dest):
        i = 1
        while os.path.exists(f"{dest}_{i}"):
            i += 1
        dest = f"{dest}_{i}"
    if not dry_run:
        os.makedirs(legacy_dir, exist_ok=True)
        shutil.move(root, dest)
    return dest


def remove_legacy_duplicate_aggregate_pdfs(output_dir: str, agg_stem: str) -> None:
    # If agg_stem is already "..._experiment_means", old behavior could also
    # emit "..._experiment_means_experiment_means_*". Remove those duplicates.
    legacy_stem = f"{agg_stem}_experiment_means"
    duplicate_pdf_paths = [
        os.path.join(output_dir, f"{legacy_stem}_datasets.pdf"),
        os.path.join(output_dir, f"{legacy_stem}_ablation.pdf"),
    ]
    for path in duplicate_pdf_paths:
        if os.path.isfile(path):
            os.remove(path)


def main() -> None:
    args = parse_args()
    requested_folds = [item.strip() for item in args.folds.split(",") if item.strip()]
    if not requested_folds:
        raise ValueError("No folds were provided.")

    input_roots = parse_input_roots(args.input_root)
    if not input_roots:
        raise ValueError("No input roots were provided.")

    os.makedirs(args.output_dir, exist_ok=True)
    dump_dir = os.path.join(args.output_dir, "dump")
    dump_csv_dir = os.path.join(dump_dir, "csv")
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(dump_csv_dir, exist_ok=True)
    move_non_pdf_files_to_dump(args.output_dir, dump_dir)
    target_roots: List[str] = []
    root_base_map: Dict[str, str] = {}
    for input_root in input_roots:
        discovered = discover_target_roots(input_root, requested_folds, args.input_name)
        for root in discovered:
            normalized_root = os.path.abspath(os.path.normpath(root))
            if normalized_root in root_base_map:
                continue
            root_base_map[normalized_root] = input_root
            target_roots.append(normalized_root)
    experiment_mean_rows: List[Dict[str, object]] = []
    root_summaries: List[Dict[str, object]] = []

    for root in target_roots:
        base_root = root_base_map.get(root, args.input_root)
        if not root_has_completed_final_results(root, requested_folds, args.input_name):
            display_name = root_display_name(root, base_root)
            print(
                f"[WARN] {root}: no final_results_*.csv found for requested folds; "
                f"skipping unfinished experiment ({display_name})."
            )
            continue

        folds = resolve_folds_for_root(root, requested_folds, args.input_name)
        fold_rows, mean_row, std_row = aggregate_root(root, folds, args.input_name)
        experiment_root_name = os.path.basename(os.path.normpath(root))
        display_name = root_display_name(root, base_root)
        scope_name, scope_fold_count = extract_scope_info(root)
        runtime_config = load_runtime_config(root)
        root_summaries.append(
            {
                "root": root,
                "root_name": display_name,
                "folds": folds,
                "fold_rows": fold_rows,
                "mean_row": mean_row,
                "std_row": std_row,
            }
        )
        exp_meta = resolve_experiment_metadata(root)
        if should_exclude_ambiguous_scaled_sum_row(exp_meta):
            print(
                f"[WARN] {root}: ambiguous scaled_sum/A run without explicit rs suffix; "
                "excluding from aggregate outputs."
            )
            continue
        comparison_model_name = extract_comparison_model_name(root, base_root)
        experiment_mean_rows.append(
            {
                "root": root,
                "model_name": comparison_model_name if comparison_model_name is not None else "dconnnet",
                "dataset": extract_dataset_name(root, base_root, runtime_config=runtime_config),
                "experiment_source": experiment_root_name,
                "experiment": exp_meta["experiment"],
                "label_mode": exp_meta["label_mode"],
                "conn_num": exp_meta["conn_num"],
                "conn_layout": exp_meta["conn_layout"],
                "conn_fusion": exp_meta["conn_fusion"],
                "fusion_branches": exp_meta.get("fusion_branches", "none"),
                "fusion_residual_scale": exp_meta.get("fusion_residual_scale"),
                "decoder_fusion": exp_meta.get("decoder_fusion", "none"),
                "seg_aux_weight": exp_meta.get("seg_aux_weight"),
                "seg_aux_variant": exp_meta.get("seg_aux_variant", "none"),
                "conn_aux_c3_weight": exp_meta.get("conn_aux_c3_weight"),
                "conn_aux_c5_weight": exp_meta.get("conn_aux_c5_weight"),
                "final_decision": str(
                    runtime_config.get("final_decision", "max")
                    if runtime_config is not None else "max"
                ),
                "loss": exp_meta["loss"],
                "fold_scope": scope_name,
                "fold_scope_count": scope_fold_count,
                "num_folds": float(len(folds)),
                "best_dice": mean_row["best_dice"],
                "best_dice_std": std_row["best_dice"],
                "best_jac": mean_row["best_jac"],
                "best_jac_std": std_row["best_jac"],
                "best_cldice": mean_row["best_cldice"],
                "best_cldice_std": std_row["best_cldice"],
                "best_precision": mean_row["best_precision"],
                "best_precision_std": std_row["best_precision"],
                "best_accuracy": mean_row["best_accuracy"],
                "best_accuracy_std": std_row["best_accuracy"],
                "best_betti_error_0": mean_row["best_betti_error_0"],
                "best_betti_error_0_std": std_row["best_betti_error_0"],
                "best_betti_error_1": mean_row["best_betti_error_1"],
                "best_betti_error_1_std": std_row["best_betti_error_1"],
                "best_hd95": mean_row.get("best_hd95", float("nan")),
                "best_hd95_std": std_row.get("best_hd95", float("nan")),
                "train_elapsed_seconds": mean_row["train_elapsed_seconds"],
                "train_elapsed_hms": mean_row["train_elapsed_hms"],
                "train_elapsed_std_seconds": std_row["train_elapsed_seconds"],
                "train_elapsed_std_hms": std_row["train_elapsed_hms"],
            }
        )

        if len(target_roots) == 1:
            output_stem = args.output_stem
        else:
            output_label = root_output_label(root, base_root)
            if len(input_roots) > 1:
                base_label = sanitize_scope_name_for_filename(os.path.basename(os.path.normpath(base_root)))
                output_label = f"{base_label}_{output_label}"
            output_stem = f"{output_label}_{args.output_stem}"

        csv_out = os.path.join(dump_csv_dir, f"{output_stem}.csv")

        write_summary_csv(csv_out, fold_rows, mean_row, std_row)

    if len(target_roots) == 1:
        maybe_write_sample_visualization(
            dump_dir,
            args.output_dir,
            args.output_stem,
            root_summaries,
            args.sample_vis_count,
            args.max_vis_models,
        )

    _moved_to_legacy: Set[str] = set()

    if len(experiment_mean_rows) >= 1:
        # Logical deduplication: when two folders parse to the same experiment
        # configuration (same dataset, fusion, branches, loss, segaux weight, etc.),
        # keep only the result from the most recently modified folder.
        _logical_seen: Dict[Tuple, Dict[str, object]] = {}
        for row in experiment_mean_rows:
            if row.get("model_name") != "dconnnet":
                continue
            lkey = _logical_experiment_key(row)
            existing = _logical_seen.get(lkey)
            if existing is None:
                _logical_seen[lkey] = row
            else:
                try:
                    row_mtime = os.path.getmtime(str(row["root"]))
                    existing_mtime = os.path.getmtime(str(existing["root"]))
                except OSError:
                    row_mtime = existing_mtime = 0.0

                # Prefer the folder with valid CSV data; fall back to mtime.
                row_valid = _root_has_valid_csv(str(row["root"]), requested_folds, args.input_name)
                existing_valid = _root_has_valid_csv(str(existing["root"]), requested_folds, args.input_name)
                prefer_row = (
                    (row_valid and not existing_valid)
                    or (row_valid == existing_valid and row_mtime > existing_mtime)
                )

                if prefer_row:
                    dropped_root = str(existing["root"])
                    kept_name = os.path.basename(str(row["root"]))
                    dropped_name = os.path.basename(dropped_root)
                    print(
                        f"[INFO] Logical duplicate: keeping newer {kept_name!r}, "
                        f"dropping older {dropped_name!r}."
                    )
                    if args.move_older_to_legacy and dropped_root not in _moved_to_legacy:
                        dest = move_experiment_to_legacy(dropped_root)
                        _moved_to_legacy.add(dropped_root)
                        print(f"[INFO] Moved {dropped_name!r} → {dest}")
                    _logical_seen[lkey] = row
                else:
                    dropped_root = str(row["root"])
                    kept_name = os.path.basename(str(existing["root"]))
                    dropped_name = os.path.basename(dropped_root)
                    print(
                        f"[INFO] Logical duplicate: keeping newer {kept_name!r}, "
                        f"dropping older {dropped_name!r}."
                    )
                    if args.move_older_to_legacy and dropped_root not in _moved_to_legacy:
                        dest = move_experiment_to_legacy(dropped_root)
                        _moved_to_legacy.add(dropped_root)
                        print(f"[INFO] Moved {dropped_name!r} → {dest}")
        _logical_kept = {id(r) for r in _logical_seen.values()}
        experiment_mean_rows = [
            r for r in experiment_mean_rows
            if r.get("model_name") != "dconnnet" or id(r) in _logical_kept
        ]

        # Deduplicate DconnNet and comparison rows separately so that two comparison
        # models with identical experiment-directory basenames on the same dataset
        # do not collide and silently drop each other.
        unique_dconnnet: Dict[Tuple, Dict[str, object]] = {}
        unique_comparison: Dict[Tuple, Dict[str, object]] = {}
        for row in experiment_mean_rows:
            if row.get("model_name") == "dconnnet":
                key = _experiment_mean_unique_key(row)
                unique_dconnnet[key] = row
            else:
                key = (
                    row.get("model_name"),
                    row.get("dataset"),
                    str(row.get("experiment_source", "")),
                )
                unique_comparison[key] = row

        dconnnet_rows = sorted(unique_dconnnet.values(), key=experiment_sort_key)
        comparison_rows = list(unique_comparison.values())

        aggregate_rows = filter_rows_for_aggregate_scope(
            dconnnet_rows,
            include_all=args.all,
        )
        if len(aggregate_rows) == 0:
            scope_label = "all counted datasets" if args.all else "default dataset scope"
            print(f"[WARN] No aggregate rows matched {scope_label}; skipping combined DconnNet outputs.")
        else:
            agg_stem = aggregate_output_stem(args.output_stem, include_all=args.all)
            agg_csv_out = os.path.join(dump_csv_dir, f"{agg_stem}.csv")
            agg_tex_out = os.path.join(dump_dir, f"{agg_stem}.tex")
            log_prefix = "[ALL]" if args.all else "[DEFAULT]"

            write_experiment_mean_csv(agg_csv_out, aggregate_rows)
            write_experiment_mean_latex(
                agg_tex_out,
                "Cross-experiment Mean Summary",
                aggregate_rows,
            )

            print(f"{log_prefix} CSV: {agg_csv_out}")
            print(f"{log_prefix} LaTeX: {agg_tex_out}")

            rows_by_dataset: Dict[str, List[Dict[str, object]]] = {}
            for row in aggregate_rows:
                dataset_name = str(row.get("dataset", "unknown_dataset"))
                if not _is_counted_dataset_name(dataset_name):
                    continue
                rows_by_dataset.setdefault(dataset_name, []).append(row)

            for dataset_name, dataset_rows in sorted(
                rows_by_dataset.items(),
                key=lambda item: _dataset_group_sort_key(item[0]),
            ):
                dataset_suffix = sanitize_scope_name_for_filename(dataset_name)
                dataset_csv_out = os.path.join(dump_csv_dir, f"{agg_stem}_{dataset_suffix}.csv")
                write_experiment_mean_csv(dataset_csv_out, dataset_rows)
                print(f"{log_prefix}:{dataset_name} CSV: {dataset_csv_out}")

            if len(rows_by_dataset) >= 1:
                _primary_dataset_names = {"drive", "deepcrack"}
                rows_by_dataset_primary = {
                    k: v for k, v in rows_by_dataset.items()
                    if _normalize_dataset_name(k) in _primary_dataset_names
                }
                rows_by_dataset_others = {
                    k: v for k, v in rows_by_dataset.items()
                    if _normalize_dataset_name(k) not in _primary_dataset_names
                }

                dataset_tex_out = os.path.join(dump_dir, f"{agg_stem}_datasets.tex")
                dataset_pdf_out = os.path.join(args.output_dir, f"{agg_stem}_datasets.pdf")
                write_experiment_mean_dataset_tables_latex(
                    dataset_tex_out,
                    "Cross-experiment Mean Summary by Dataset",
                    rows_by_dataset_primary if rows_by_dataset_primary else rows_by_dataset,
                )
                build_pdf(dataset_tex_out, dataset_pdf_out)
                print(f"{log_prefix}:DATASETS LaTeX: {dataset_tex_out}")
                print(f"{log_prefix}:DATASETS PDF: {dataset_pdf_out}")

                if rows_by_dataset_others:
                    dataset_others_tex_out = os.path.join(dump_dir, f"{agg_stem}_datasets_others.tex")
                    dataset_others_pdf_out = os.path.join(args.output_dir, f"{agg_stem}_datasets_others.pdf")
                    write_experiment_mean_dataset_tables_latex(
                        dataset_others_tex_out,
                        "Cross-experiment Mean Summary by Dataset (Others)",
                        rows_by_dataset_others,
                    )
                    build_pdf(dataset_others_tex_out, dataset_others_pdf_out)
                    print(f"{log_prefix}:DATASETS_OTHERS LaTeX: {dataset_others_tex_out}")
                    print(f"{log_prefix}:DATASETS_OTHERS PDF: {dataset_others_pdf_out}")

                ablation_tex_out = os.path.join(dump_dir, f"{agg_stem}_ablation.tex")
                ablation_pdf_out = os.path.join(args.output_dir, f"{agg_stem}_ablation.pdf")
                ablation_paths: Dict[str, object] = {}
                write_ablation_latex(
                    ablation_tex_out,
                    aggregate_rows,
                    include_buckets={"DRIVE", "DeepCrack"},
                    paths_out=ablation_paths,
                )
                build_pdf(ablation_tex_out, ablation_pdf_out)
                summary_ablation_tex_out = os.path.join(args.output_dir, f"{agg_stem}_ablation.tex")
                shutil.copyfile(ablation_tex_out, summary_ablation_tex_out)
                ablation_paths_json_out = os.path.join(args.output_dir, f"{agg_stem}_ablation_paths.json")
                with open(ablation_paths_json_out, "w") as _f:
                    json.dump(ablation_paths, _f, indent=2, default=str)
                print(f"{log_prefix}:ABLATION PATHS JSON: {ablation_paths_json_out}")

                ablation_others_tex_out = os.path.join(dump_dir, f"{agg_stem}_ablation_others.tex")
                ablation_others_pdf_out = os.path.join(args.output_dir, f"{agg_stem}_ablation_others.pdf")
                write_ablation_latex(
                    ablation_others_tex_out,
                    aggregate_rows,
                    include_buckets={"CHASE", "OCTA6M", "OCTA3M", "Others"},
                )
                build_pdf(ablation_others_tex_out, ablation_others_pdf_out)

                print(f"{log_prefix}:ABLATION LaTeX: {ablation_tex_out}")
                print(f"{log_prefix}:ABLATION LaTeX (summary root): {summary_ablation_tex_out}")
                print(f"{log_prefix}:ABLATION PDF: {ablation_pdf_out}")
                print(f"{log_prefix}:ABLATION LaTeX (others): {ablation_others_tex_out}")
                print(f"{log_prefix}:ABLATION PDF (others): {ablation_others_pdf_out}")

                remove_legacy_duplicate_aggregate_pdfs(args.output_dir, agg_stem)

        if comparison_rows:
            scoped_comparison_rows = filter_rows_for_aggregate_scope(
                comparison_rows,
                include_all=args.all,
            )
            comp_rows_by_dataset: Dict[str, List[Dict[str, object]]] = {}
            for row in scoped_comparison_rows:
                dataset_name = str(row.get("dataset", "unknown_dataset"))
                if _is_counted_dataset_name(dataset_name):
                    comp_rows_by_dataset.setdefault(dataset_name, []).append(row)

            if comp_rows_by_dataset:
                comp_stem = f"{args.output_stem}_comparison_models"
                comp_csv_out = os.path.join(dump_csv_dir, f"{comp_stem}.csv")
                comp_tex_out = os.path.join(dump_dir, f"{comp_stem}.tex")
                comp_pdf_out = os.path.join(args.output_dir, f"{comp_stem}.pdf")
                write_experiment_mean_csv(comp_csv_out, scoped_comparison_rows)
                write_comparison_models_latex(comp_tex_out, comp_rows_by_dataset)
                build_pdf(comp_tex_out, comp_pdf_out)
                print(f"[COMPARISON] CSV: {comp_csv_out}")
                print(f"[COMPARISON] PDF: {comp_pdf_out}")
            else:
                scope_label = "all counted datasets" if args.all else "default dataset scope"
                print(f"[WARN] No comparison model rows matched {scope_label}; skipping comparison PDF.")


if __name__ == "__main__":
    main()
