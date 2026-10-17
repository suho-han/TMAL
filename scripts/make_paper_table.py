#!/usr/bin/env python3
"""
Generate the per-sample qualitative figures for the paper.
Supports DRIVE, DeepCrack, CHASE, OCTA500-3M, OCTA500-6M, XACV, FIVES.

Writes per-dataset prediction panels and JSON under output/paper_tables/figures/,
which scripts/generate_figure.py then renders into the final figure PDFs/PNGs.

Usage:
    uv run scripts/make_paper_table.py                        # all datasets
    uv run scripts/make_paper_table.py --datasets DRIVE CHASE # subset
    uv run scripts/make_paper_table.py --output-dir output/paper_tables
    uv run scripts/make_paper_table.py --input-root output,output_dropone_svariation
    uv run scripts/make_paper_table.py --list DeepCrack       # print sorted experiment list
    uv run scripts/make_paper_table.py --ablation             # also export ablation-table
                                                                # row panels (Tables 1/3/5-style:
                                                                # category-grouped, S-set, branch
                                                                # composition), sourced from
                                                                # summary_experiment_means_ablation_paths.json
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from aggregate_results import (_experiment_mean_unique_key, _loss_sort_key, aggregate_root, checkpoint_image_path, discover_target_roots, experiment_sort_key, extract_dataset_name, extract_scope_info,
                               load_panel_image, load_runtime_config, parse_input_roots, resolve_experiment_metadata, resolve_folds_for_root, should_exclude_ambiguous_scaled_sum_row)
from PIL import Image as PilImage
from PIL import ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))

# ── Configuration (edit here) ─────────────────────────────────────────────────

# Per-dataset configuration.
# experiments: list of (query, display_name) pairs, in display order.
#   query is either:
#     str  — exact match on experiment_source (output subdirectory name)
#     dict — field-based match; all key/value pairs must equal the exp row fields
#   Run `--list <DATASET>` to print the sorted experiment list with field values.
# external_baselines: display name → directory containing final_results*.csv.

_DCONNNET_QUERY = "binary_8_bce"
# Fixed paper-table Proposed anchor. Keep final_decision=max so fd ablations
# remain separate experiments instead of silently replacing the baseline row.
_PROPOSED_QUERY = {"label_mode": "dist_inverted", "conn_num": 8, "conn_fusion": "dg_dsf",
                   "fusion_branches": "io",
                   "loss": "smooth_l1", "seg_aux_weight": 0.5,
                   "conn_aux_c3_weight": 0.5, "conn_aux_c5_weight": 0.3,
                   "final_decision": "max"}
_DSF_QUERY    = {"label_mode": "dist_inverted", "conn_num": 8, "conn_fusion": "dsf",
                 "fusion_branches": "io", "loss": "smooth_l1", "seg_aux_weight": 0.5,
                 "final_decision": "max"}
_DG_DSF_QUERY = {"label_mode": "dist_inverted", "conn_num": 8, "conn_fusion": "dg_dsf",
                 "fusion_branches": "io", "loss": "smooth_l1", "seg_aux_weight": 0.5,
                 "final_decision": "max"}

DATASET_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "DRIVE",
        "root": "output/drive",
        "folds": ["1"],
        "input_name": "final_results_{fold}.csv",
        "experiments": [
            (_DCONNNET_QUERY, "DconnNet"),
            (_DSF_QUERY,      "DSF"),
            (_DG_DSF_QUERY,   "DG-DSF"),
            (_PROPOSED_QUERY, "Proposed"),
        ],
        "external_baselines": {
            "U-Net":           "output/baseline/drive/unet",
            "Attention U-Net": "output/baseline/drive/attention_unet",
            "FRU-Net":         "output/baseline/drive/frunet",
            "CSNet":           "output/baseline/drive/csnet",
            "clDice":          "output/baseline/drive/cldice",
            "CAPE":            "output/baseline/drive/cape",
            "VGAN":            "output/baseline/drive/VGAN",
            "SegDiff":         "output/baseline/drive/segdiff",
            "MedSegDiff":      "output/baseline/drive/medsegdiff",
        },
        "source_data": {
            "data_root":      "data/DRIVE/test",
            "image_subdir":   "images",
            "gt_subdir":      "masks",
            "image_name_fmt": "{sample_name}.tif",
            "gt_name_fmt":    "{sample_name}.gif",
        },
    },
    {
        "name": "DeepCrack",
        "root": "output/deepcrack",
        "folds": ["1"],
        "input_name": "final_results_{fold}.csv",
        "experiments": [
            (_DCONNNET_QUERY, "DconnNet"),
            (_DSF_QUERY,      "DSF"),
            (_DG_DSF_QUERY,   "DG-DSF"),
            (_PROPOSED_QUERY, "Proposed"),
        ],
        "external_baselines": {
            "U-Net":           "output/baseline/deepcrack/unet",
            "Attention U-Net": "output/baseline/deepcrack/attention_unet",
            "FRU-Net":         "output/baseline/deepcrack/frunet",
            "CSNet":           "output/baseline/deepcrack/csnet",
            "clDice":          "output/baseline/deepcrack/cldice",
            "VGAN":            "output/baseline/deepcrack/VGAN",
            "CAPE":            "output/baseline/deepcrack/cape",
            "SegDiff":         "output/baseline/deepcrack/segdiff",
            "MedSegDiff":      "output/baseline/deepcrack/medsegdiff",
        },
        "source_data": {
            "data_root":      "data/DeepCrack",
            "image_subdir":   "test_img",
            "gt_subdir":      "test_lab",
            "image_name_fmt": "{sample_name}.jpg",
            "gt_name_fmt":    "{sample_name}.png",
        },
        # Rendered in addition to the auto-picked highest-Dice sample.
        "extra_sample_ids": ["11302-1", "11302-2"],
    },
    {
        "name": "CHASE",
        "root": "output/chase",
        "folds": ["1"],
        "input_name": "final_results_{fold}.csv",
        "experiments": [
            (_DCONNNET_QUERY, "DconnNet"),
            (_DSF_QUERY,      "DSF"),
            (_DG_DSF_QUERY,   "DG-DSF"),
            (_PROPOSED_QUERY, "Proposed"),
        ],
        "external_baselines": {
            "U-Net":           "output/baseline/chase/unet",
            "Attention U-Net": "output/baseline/chase/attention_unet",
            "FRU-Net":         "output/baseline/chase/frunet",
            "CSNet":           "output/baseline/chase/csnet",
        },
        "source_data": {
            "data_root":      "data/chase",
            "image_subdir":   "images",
            "gt_subdir":      "gt",
            "image_name_fmt": "Image_{sample_name}.jpg",
            "gt_name_fmt":    "Image_{sample_name}_1stHO.png",
        },
    },
    {
        "name": "OCTA500-3M",
        "root": "output/octa500-3M",
        "folds": ["1"],
        "input_name": "final_results_{fold}.csv",
        "experiments": [
            (_DCONNNET_QUERY, "DconnNet"),
            (_DSF_QUERY,      "DSF"),
            (_DG_DSF_QUERY,   "DG-DSF"),
            (_PROPOSED_QUERY, "Proposed"),
        ],
        "external_baselines": {
            "U-Net":           "output/baseline/octa500-3M/unet",
            "Attention U-Net": "output/baseline/octa500-3M/attention_unet",
            "FRU-Net":         "output/baseline/octa500-3M/frunet",
            "CSNet":           "output/baseline/octa500-3M/csnet",
        },
        "source_data": {
            "data_root":      "data/OCTA500_3M/test",
            "image_subdir":   "images",
            "gt_subdir":      "labels",
            "image_name_fmt": "{sample_name}.bmp",
            "gt_name_fmt":    "{sample_name}.bmp",
        },
    },
    {
        "name": "OCTA500-6M",
        "root": "output/octa500-6M",
        "folds": ["1"],
        "input_name": "final_results_{fold}.csv",
        "experiments": [
            (_DCONNNET_QUERY, "DconnNet"),
            (_DSF_QUERY,      "DSF"),
            (_DG_DSF_QUERY,   "DG-DSF"),
            (_PROPOSED_QUERY, "Proposed"),
        ],
        "external_baselines": {
            "U-Net":           "output/baseline/octa500-6M/unet",
            "Attention U-Net": "output/baseline/octa500-6M/attention_unet",
            "FRU-Net":         "output/baseline/octa500-6M/frunet",
            "CSNet":           "output/baseline/octa500-6M/csnet",
        },
        "source_data": {
            "data_root":      "data/OCTA500_6M/test",
            "image_subdir":   "images",
            "gt_subdir":      "labels",
            "image_name_fmt": "{sample_name}.bmp",
            "gt_name_fmt":    "{sample_name}.bmp",
        },
    },
    {
        "name": "XACV",
        "root": "output/xacv",
        "folds": ["1"],
        "input_name": "final_results_{fold}.csv",
        "experiments": [
            (_DCONNNET_QUERY, "DconnNet"),
            (_DSF_QUERY,      "DSF"),
            (_DG_DSF_QUERY,   "DG-DSF"),
            (_PROPOSED_QUERY, "Proposed"),
        ],
        "external_baselines": {},
    },
    {
        "name": "FIVES",
        "root": "output/fives",
        "folds": ["1"],
        "input_name": "final_results_{fold}.csv",
        "experiments": [
            (_DCONNNET_QUERY, "DconnNet"),
            (_DSF_QUERY,      "DSF"),
            (_DG_DSF_QUERY,   "DG-DSF"),
            (_PROPOSED_QUERY, "Proposed"),
        ],
        "external_baselines": {},
        "source_data": {
            "data_root":      "data/FIVES/test",
            "image_subdir":   "Original",
            "gt_subdir":      "Ground truth",
            "image_name_fmt": "{sample_name}.png",
            "gt_name_fmt":    "{sample_name}.png",
        },
    },
]

OUTPUT_DIR = "output/paper_tables"
DEFAULT_DATASETS = ["DRIVE", "DeepCrack", "CHASE", "OCTA500-3M", "OCTA500-6M", "XACV", "FIVES"]

# ── Data loading ──────────────────────────────────────────────────────────────


def resolve_dataset_roots(dataset_root: str, input_roots: Optional[List[str]]) -> List[str]:
    dataset_name = os.path.basename(os.path.normpath(dataset_root))
    if not input_roots:
        return [os.path.abspath(os.path.normpath(dataset_root))]
    resolved: List[str] = []
    seen: Set[str] = set()
    for base_root in input_roots:
        base_norm = os.path.abspath(os.path.normpath(base_root))
        if os.path.basename(base_norm) == dataset_name:
            candidate = base_norm
        else:
            candidate = os.path.join(base_norm, dataset_name)
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return resolved


def load_dataset_experiments(cfg: Dict[str, Any], input_roots: Optional[List[str]] = None) -> List[Dict]:
    """Return experiments for one dataset sorted the same way as the datasets PDF."""
    roots = resolve_dataset_roots(cfg["root"], input_roots)
    folds = cfg["folds"]
    input_name = cfg["input_name"]

    target_roots: List[str] = []
    root_base_map: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            print(f"[WARN] Dataset root not found: {root}")
            continue
        discovered = discover_target_roots(root, folds, input_name)
        for exp_root in discovered:
            normalized_root = os.path.abspath(os.path.normpath(exp_root))
            if normalized_root in root_base_map:
                continue
            root_base_map[normalized_root] = root
            target_roots.append(normalized_root)

    if not target_roots:
        return []

    rows = []
    for exp_root in target_roots:
        base_root = root_base_map.get(exp_root, cfg["root"])
        resolved_folds = resolve_folds_for_root(exp_root, folds, input_name)
        _, mean_row, std_row = aggregate_root(exp_root, resolved_folds, input_name)
        exp_root_name = os.path.basename(os.path.normpath(exp_root))
        scope_name, scope_fold_count = extract_scope_info(exp_root)
        runtime_config = load_runtime_config(exp_root)
        exp_meta = resolve_experiment_metadata(exp_root)
        if should_exclude_ambiguous_scaled_sum_row(exp_meta):
            continue
        rows.append({
            "experiment_root":      exp_root,
            "dataset":               extract_dataset_name(exp_root, base_root, runtime_config=runtime_config),
            "experiment_source":     exp_root_name,
            "experiment":            exp_meta["experiment"],
            "label_mode":            exp_meta["label_mode"],
            "conn_num":              exp_meta["conn_num"],
            "conn_layout":           exp_meta["conn_layout"],
            "conn_fusion":           exp_meta["conn_fusion"],
            "fusion_branches":       exp_meta.get("fusion_branches", "none"),
            "fusion_residual_scale": exp_meta.get("fusion_residual_scale"),
            "decoder_fusion":        exp_meta.get("decoder_fusion", "none"),
            "seg_aux_weight":        exp_meta.get("seg_aux_weight"),
            "seg_aux_variant":       exp_meta.get("seg_aux_variant", "none"),
            "conn_aux_c3_weight":    exp_meta.get("conn_aux_c3_weight"),
            "conn_aux_c5_weight":    exp_meta.get("conn_aux_c5_weight"),
            "final_decision":        str(
                runtime_config.get("final_decision", "max")
                if runtime_config is not None else "max"
            ),
            "loss":                  exp_meta["loss"],
            "fold_scope":            scope_name,
            "fold_scope_count":      scope_fold_count,
            "num_folds":             float(len(resolved_folds)),
            "dice":                  mean_row["best_dice"],
            "jac":                   mean_row["best_jac"],
            "cldice":                mean_row["best_cldice"],
            "betti_error_0":         mean_row["best_betti_error_0"],
            "betti_error_1":         mean_row["best_betti_error_1"],
            "hd95":                  mean_row.get("best_hd95", float("nan")),
            "dice_std":              std_row["best_dice"],
            "jac_std":               std_row["best_jac"],
            "cldice_std":            std_row["best_cldice"],
            "betti_error_0_std":     std_row["best_betti_error_0"],
            "betti_error_1_std":     std_row["best_betti_error_1"],
            "hd95_std":              std_row.get("best_hd95", float("nan")),
        })

    unique: Dict = {}
    for row in rows:
        key = _experiment_mean_unique_key(row)
        unique[key] = row
    rows = sorted(unique.values(), key=experiment_sort_key)

    rows_by_loss: Dict[str, List] = {}
    for row in rows:
        rows_by_loss.setdefault(row["loss"], []).append(row)
    ordered: List = []
    for loss in sorted(rows_by_loss, key=_loss_sort_key):
        ordered.extend(rows_by_loss[loss])
    return ordered


def _match_exp(exp: Dict, query) -> bool:
    if isinstance(query, str):
        return exp["experiment_source"] == query
    return all(str(exp.get(k, "")) == str(v) for k, v in query.items())


def _find_experiment(exp_list: List[Dict], query) -> Optional[Dict]:
    matches = [exp for exp in exp_list if _match_exp(exp, query)]
    if not matches:
        return None
    if len(matches) > 1:
        best = max(matches, key=lambda e: float(e.get("dice", float("-inf"))))
        sources = [e["experiment_source"] for e in matches]
        print(f"[INFO] Query {query!r} matched {len(matches)} experiments "
              f"{sources}; using best Dice: {best['experiment_source']}.")
        return best
    return matches[0]


def list_dataset_experiments(cfg: Dict[str, Any], input_roots: Optional[List[str]] = None) -> None:
    """Print the sorted experiment list with key fields (for building experiment queries)."""
    exp_list = load_dataset_experiments(cfg, input_roots=input_roots)
    print(f"\n[LIST] {cfg['name']} — {len(exp_list)} experiments (sorted order):")
    fields = ("label_mode", "conn_num", "conn_fusion", "conn_layout",
              "loss", "seg_aux_weight", "conn_aux_c3_weight", "conn_aux_c5_weight",
              "final_decision")
    header = f"  {'#':>3}  {'experiment_source':<55}  " + "  ".join(f"{f}" for f in fields)
    print(header)
    for i, exp in enumerate(exp_list, 1):
        vals = "  ".join(f"{str(exp.get(f, '')):<{max(len(f),4)}}" for f in fields)
        print(f"  {i:3d}  {exp['experiment_source']:<55}  {vals}  dice={exp['dice']:.4f}")


# ── Figure / JSON generation ──────────────────────────────────────────────────

def _model_dir(exp_root: str, fold: str) -> str:
    candidate = os.path.join(exp_root, "models", fold)
    return candidate if os.path.isdir(candidate) else os.path.join(exp_root, "models")


def _dominant_epoch(sample_rows: List[Dict]) -> Optional[int]:
    if not sample_rows:
        return None
    return max(int(r["epoch"]) for r in sample_rows)


def _save_jpg(arr, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = PilImage.fromarray(arr).convert("RGB")
    img.save(path, format="JPEG", quality=95)


def _clean_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


_SAMPLE_METRIC_KEYS = (
    "val_loss", "dice", "jac", "cldice",
    "precision", "accuracy", "betti_error_0", "betti_error_1", "hd95",
)

_FIGURE_PANEL_TOKENS = {
    "VGAN": "v-gan",
}


def _figure_panel_token(display_name: str) -> str:
    return _FIGURE_PANEL_TOKENS.get(display_name, display_name.lower().replace(" ", "_"))


def _normalize_sample_name(sample_name: object) -> str:
    return os.path.splitext(str(sample_name).strip())[0]


def _sample_name_variants(sample_name: object) -> List[str]:
    base = _normalize_sample_name(sample_name)
    variants = [base]
    if base.isdigit():
        number = int(base)
        variants.extend([str(number), f"{number:02d}", f"{number:03d}"])
    ordered: List[str] = []
    seen: Set[str] = set()
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            ordered.append(variant)
    return ordered


def _sorted_filename_stems(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []

    def sort_key(name: str) -> List[Tuple[int, object]]:
        stem = os.path.splitext(name)[0]
        parts = re.split(r"(\d+)", stem)
        return [(0, int(part)) if part.isdigit() else (1, part.lower()) for part in parts if part]

    names = [entry.name for entry in os.scandir(directory) if entry.is_file()]
    return [os.path.splitext(name)[0] for name in sorted(names, key=sort_key)]


def _build_source_sample_order(source_data: Optional[Dict[str, Any]]) -> List[str]:
    if not source_data:
        return []
    image_dir = os.path.join(source_data["data_root"], source_data["image_subdir"])
    return _sorted_filename_stems(image_dir)


def _source_sample_name_from_batch(source_sample_order: List[str], batch: Optional[int]) -> Optional[str]:
    if batch is None:
        return None
    if 0 <= batch < len(source_sample_order):
        return source_sample_order[batch]
    return None


def _parse_batch_value(raw_value: object) -> Optional[int]:
    text = str(raw_value).strip()
    if not text:
        return None
    match = re.search(r"batch_(\d+)", text, re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    return int(text) if text.isdigit() else None


def _csv_sample_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row_type = str(raw_row.get("result_type", raw_row.get("type", "sample"))).strip().lower()
            sample_name = (
                raw_row.get("sample_name")
                or raw_row.get("sample")
                or raw_row.get("evaluated_split")
                or raw_row.get("filename")
            )
            if row_type not in {"sample", "", "pred"}:
                continue
            if sample_name is None:
                continue
            sample_name = _normalize_sample_name(sample_name)
            if sample_name.upper() in {"MEAN", "STD", "FINAL"}:
                continue

            row: Dict[str, Any] = {"sample_name": sample_name}
            if str(raw_row.get("epoch", "")).strip().isdigit():
                row["epoch"] = int(raw_row["epoch"])
            if str(raw_row.get("batch", "")).strip().isdigit():
                row["batch"] = int(raw_row["batch"])
            elif raw_row.get("checkpoint"):
                batch = _parse_batch_value(raw_row.get("checkpoint"))
                if batch is not None:
                    row["batch"] = batch
            if str(raw_row.get("eval_epoch", "")).strip().isdigit() and "epoch" not in row:
                row["epoch"] = int(raw_row["eval_epoch"])
            if str(raw_row.get("source_pred_epoch", "")).strip().isdigit() and "epoch" not in row:
                row["epoch"] = int(raw_row["source_pred_epoch"])

            metric_map = {
                "dice": ("dice",),
                "jac": ("jac", "iou"),
                "cldice": ("cldice",),
                "betti_error_0": ("betti_error_0", "betti0_error"),
                "betti_error_1": ("betti_error_1", "betti1_error"),
                "hd95": ("hd95",),
                "val_loss": ("val_loss",),
                "precision": ("precision",),
                "accuracy": ("accuracy",),
            }
            for key, aliases in metric_map.items():
                for alias in aliases:
                    if alias in raw_row and raw_row[alias] not in {None, ""}:
                        row[key] = _clean_float(raw_row[alias])
                        break
            rows.append(row)
    return rows


def _candidate_figure_csv_paths(model_dir: str, source_root: str, dataset_slug: str) -> List[str]:
    candidates = [
        os.path.join(model_dir, "test_sample_metrics.csv"),
        os.path.join(source_root, "test_sample_metrics.csv"),
        os.path.join(source_root, "models", "test_sample_metrics.csv"),
        os.path.join(source_root, f"metrics_{dataset_slug}.csv"),
        os.path.join(source_root, f"results_{dataset_slug}.csv"),
        os.path.join(source_root, "metrics.csv"),
        os.path.join(source_root, "final_results.csv"),
    ]
    ordered: List[str] = []
    seen: Set[str] = set()
    for path in candidates:
        if path not in seen and os.path.isfile(path):
            seen.add(path)
            ordered.append(path)
    return ordered


def _load_figure_sample_rows(model_dir: str, source_root: str, dataset_slug: str) -> List[Dict[str, Any]]:
    for path in _candidate_figure_csv_paths(model_dir, source_root, dataset_slug):
        if os.path.basename(path) == "test_sample_metrics.csv":
            return _parse_sample_metrics_full(path)
        return _csv_sample_rows(path)
    return []


def _figure_source_root(model_dir: str) -> str:
    parent = os.path.dirname(model_dir)
    if os.path.basename(parent) == "models":
        return os.path.dirname(parent)
    return parent


def _find_fallback_prediction_image(
    source_root: str,
    sample_name: object,
    batch: Optional[int] = None,
    preferred_fragments: Optional[List[str]] = None,
) -> Optional[str]:
    if not source_root or not os.path.isdir(source_root):
        return None

    target_dirs = {
        "predictions", "prediction", "preds", "pred", "results", "output", "outputs", "images", "figures"
    }
    extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def _search(root: str, allow_any_dir: bool = False) -> Optional[str]:
        for current_root, dirnames, filenames in os.walk(root):
            rel_depth = os.path.relpath(current_root, root).count(os.sep)
            if rel_depth > 5:
                dirnames[:] = []
                continue
            current_dir = os.path.basename(current_root).lower()
            if not allow_any_dir and current_dir not in target_dirs:
                continue
            if batch is not None:
                batch_variants = [f"batch_{batch}", f"batch_{batch:02d}", f"batch_{batch:03d}"]
                preferred_batch_fragments = preferred_fragments or []
                for fragment in preferred_batch_fragments:
                    for variant in batch_variants:
                        for ext in extensions:
                            candidate = os.path.join(current_root, f"{variant}{fragment}{ext}")
                            if os.path.isfile(candidate):
                                return candidate
                for variant in batch_variants:
                    for ext in extensions:
                        candidate = os.path.join(current_root, f"{variant}{ext}")
                        if os.path.isfile(candidate):
                            return candidate
            for variant in _sample_name_variants(sample_name):
                for ext in extensions:
                    candidate = os.path.join(current_root, f"{variant}{ext}")
                    if os.path.isfile(candidate):
                        return candidate
            for filename in filenames:
                stem = os.path.splitext(filename)[0].lower()
                if preferred_fragments and batch is not None:
                    batch_variants = [f"batch_{batch}", f"batch_{batch:02d}", f"batch_{batch:03d}"]
                    if any(variant in stem and any(fragment in stem for fragment in preferred_fragments) for variant in batch_variants):
                        if os.path.splitext(filename)[1].lower() in extensions:
                            return os.path.join(current_root, filename)
                if any(variant in stem for variant in _sample_name_variants(sample_name)):
                    if os.path.splitext(filename)[1].lower() in extensions:
                        return os.path.join(current_root, filename)
                if batch is not None and any(variant in stem for variant in [f"batch_{batch}", f"batch_{batch:02d}", f"batch_{batch:03d}"]):
                    if os.path.splitext(filename)[1].lower() in extensions:
                        return os.path.join(current_root, filename)
        return None

    direct = _search(source_root)
    if direct is not None:
        return direct

    relaxed = _search(source_root, allow_any_dir=True)
    if relaxed is not None:
        return relaxed

    root_parent = os.path.dirname(source_root)
    if os.path.isdir(root_parent):
        for entry in os.scandir(root_parent):
            if not entry.is_dir():
                continue
            entry_name = entry.name.lower()
            if "segdiff" not in entry_name and "medsegdiff" not in entry_name:
                continue
            candidate = _search(entry.path, allow_any_dir=True)
            if candidate is not None:
                return candidate
    return None


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _parse_sample_metrics_full(path: str) -> List[Dict[str, Any]]:
    """Read test_sample_metrics.csv preserving all numeric columns."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry: Dict[str, Any] = {
                "epoch":           int(row["epoch"]),
                "batch":           int(row["batch"]),
                "sample_in_batch": int(row["sample_in_batch"]),
                "sample_name":     row["sample_name"],
            }
            for key in _SAMPLE_METRIC_KEYS:
                if key in row:
                    entry[key] = _clean_float(row[key])
            rows.append(entry)
    return rows


def _find_source_image(data_root: str, subdir: str, name_fmt: str,
                       sample_name: str, extra_exts: Tuple[str, ...] = ()) -> Optional[str]:
    """Find an original dataset file. Tries the template name, then alternate extensions."""
    basename = name_fmt.format(sample_name=sample_name)
    path = os.path.join(data_root, subdir, basename)
    if os.path.isfile(path):
        return path
    stem = os.path.splitext(basename)[0]
    for ext in extra_exts:
        candidate = os.path.join(data_root, subdir, f"{stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
    return None


def _copy_as_jpg(src_path: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    PilImage.open(src_path).convert("RGB").save(dst_path, format="JPEG", quality=95)


def _save_concat_grid(panel_paths: List[Optional[str]], labels: List[str], out_path: str) -> None:
    pairs = [(p, l) for p, l in zip(panel_paths, labels) if p and os.path.isfile(p)]
    if not pairs:
        return
    imgs = [PilImage.open(p).convert("RGB") for p, _ in pairs]
    h = min(img.height for img in imgs)
    resized = [img.resize((max(1, img.width * h // img.height), h), PilImage.LANCZOS) for img in imgs]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    label_h = 32
    total_w = sum(img.width for img in resized)
    grid = PilImage.new("RGB", (total_w, label_h + h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid)

    x = 0
    for img, (_, label) in zip(resized, pairs):
        grid.paste(img, (x, label_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x + (img.width - text_w) // 2, (label_h - text_h) // 2), label, fill=(30, 30, 30), font=font)
        x += img.width

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path, format="JPEG", quality=95)


def generate_dataset_figure(cfg: Dict[str, Any], output_dir: str, input_roots: Optional[List[str]] = None) -> None:
    exp_list = load_dataset_experiments(cfg, input_roots=input_roots)
    if not exp_list:
        return

    models: List[Tuple[str, str]] = []
    for query, display_name in cfg["experiments"]:
        exp = _find_experiment(exp_list, query)
        if exp is None:
            print(f"[WARN] [{cfg['name']}] figure: no match for {query!r}; skipping '{display_name}'.")
            continue
        exp_root = exp.get("experiment_root") or os.path.join(cfg["root"], exp["experiment_source"])
        models.append((display_name, _model_dir(exp_root, cfg["folds"][0])))

    if not models:
        return

    ref_mdir = next((mdir for name, mdir in models if name == "Proposed"), models[-1][1])
    sample_csv = os.path.join(ref_mdir, "test_sample_metrics.csv")
    if not os.path.isfile(sample_csv):
        print(f"[WARN] [{cfg['name']}] figure: {sample_csv} not found; skipping.")
        return

    all_rows = _parse_sample_metrics_full(sample_csv)
    ref_epoch = _dominant_epoch(all_rows)
    if ref_epoch is None:
        return

    by_name: Dict[str, Dict] = {}
    for r in all_rows:
        if int(r["epoch"]) != ref_epoch:
            continue
        name = str(r["sample_name"])
        if name not in by_name or float(r["dice"]) > float(by_name[name]["dice"]):
            by_name[name] = r
    unique = list(by_name.values())

    dataset_safe = cfg["name"].lower().replace("-", "_").replace(" ", "_")

    model_data: List[Tuple[str, str, str, Optional[int], List[Dict]]] = []
    for display_name, mdir in models:
        source_root = _figure_source_root(mdir)
        rows = _load_figure_sample_rows(mdir, source_root, dataset_safe)
        epoch = _dominant_epoch(rows) if any("epoch" in row for row in rows) else None
        model_data.append((display_name, mdir, source_root, epoch, rows))
    for ext_name, ext_dir in cfg.get("external_baselines", {}).items():
        ext_mdir = _model_dir(ext_dir, cfg["folds"][0])
        rows = _load_figure_sample_rows(ext_mdir, ext_dir, dataset_safe)
        epoch = _dominant_epoch(rows) if any("epoch" in row for row in rows) else None
        model_data.append((ext_name, ext_mdir, ext_dir, epoch, rows))

    available_sample_sets: List[Set[str]] = []
    for _, _, _, model_epoch, rows in model_data:
        if model_epoch is None:
            available_sample_sets.append({str(r.get("sample_name")) for r in rows})
        else:
            available_sample_sets.append({
                str(r.get("sample_name"))
                for r in rows
                if "epoch" not in r or int(r["epoch"]) == model_epoch
            })
    common_sample_names = set.intersection(*available_sample_sets) if available_sample_sets else set()
    if common_sample_names:
        unique = [r for r in unique if str(r.get("sample_name")) in common_sample_names]

    # Keep only the single highest-Dice sample (by the Proposed model's Dice),
    # matching the sample selection in notebooks/paper_figures.ipynb. Excludes
    # the explicitly requested extra samples below from the auto-pick pool, so
    # the auto pick can't collapse onto one of them.
    extra_ids = cfg.get("extra_sample_ids") or []
    extra_ids_normalized = {_normalize_sample_name(x) for x in extra_ids}
    auto_pick_candidates = [
        r for r in unique
        if _normalize_sample_name(r.get("sample_name")) not in extra_ids_normalized
    ]
    top_samples = sorted(
        auto_pick_candidates,
        key=lambda r: (
            -float(r["dice"] or 0.0),
            str(r.get("sample_name", "")),
        ),
    )[:1]

    # Plus any explicitly requested extra samples (rendered in addition to
    # the auto-picked one, not instead of it).
    picked_names = {str(r.get("sample_name")) for r in top_samples}
    for extra_id in extra_ids:
        if extra_id in picked_names:
            continue
        # Look up extras from the reference model's full sample set (by_name),
        # not the common_sample_names-filtered `unique` -- an extra sample only
        # needs to exist for the reference (Proposed) model; baselines that are
        # missing it (e.g. a baseline evaluated on a subset of the test set)
        # simply render without that one panel.
        extra_row = next(
            (r for r in by_name.values() if _normalize_sample_name(r.get("sample_name")) == _normalize_sample_name(extra_id)),
            None,
        )
        if extra_row is None:
            print(f"[WARN] [{cfg['name']}] figure: extra sample '{extra_id}' not found; skipping.")
            continue
        top_samples.append(extra_row)
        picked_names.add(str(extra_row.get("sample_name")))

    figure_order = {
        "Proposed": 0,
        "U-Net": 1,
        "Attention U-Net": 2,
        "FRU-Net": 3,
        "CSNet": 4,
        "VGAN": 5,
        "SegDiff": 6,
        "MedSegDiff": 7,
        "clDice": 8,
        "CAPE": 9,
        "DconnNet": 10,
    }
    model_data.sort(key=lambda item: figure_order.get(item[0], len(figure_order)))

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    samples_json: Dict[str, Dict[str, Any]] = {}
    source_sample_order = _build_source_sample_order(cfg.get("source_data"))

    for ref_sample in top_samples:
        sample_name = str(ref_sample["sample_name"])
        sample_name_safe = re.sub(r"[/\\]", "_", sample_name)
        ref_batch = int(ref_sample["batch"])
        entry: Dict[str, Any] = {"id": sample_name}

        source_data = cfg.get("source_data")
        ordered_sample_name = _source_sample_name_from_batch(source_sample_order, ref_batch)
        panel_index = 1  # 1-based column index used in saved filenames

        # Input image
        dst = os.path.join(figures_dir, f"{dataset_safe}_{sample_name_safe}_{panel_index}_image.jpg")
        if source_data:
            img_path = _find_source_image(
                source_data["data_root"], source_data["image_subdir"],
                source_data["image_name_fmt"], sample_name,
            )
            if img_path:
                _copy_as_jpg(img_path, dst)
                entry["image"] = f"figures/{dataset_safe}_{sample_name_safe}_{panel_index}_image.jpg"
            else:
                print(f"[WARN] [{cfg['name']}] source image not found for sample '{sample_name}'")
        else:
            src = checkpoint_image_path(ref_mdir, ref_epoch, "image", ref_batch)
            if src:
                _save_jpg(load_panel_image(src, "image"), dst)
                entry["image"] = f"figures/{dataset_safe}_{sample_name_safe}_{panel_index}_image.jpg"
        panel_index += 1

        # GT mask
        dst = os.path.join(figures_dir, f"{dataset_safe}_{sample_name_safe}_{panel_index}_gt.jpg")
        if source_data:
            gt_path = _find_source_image(
                source_data["data_root"], source_data["gt_subdir"],
                source_data["gt_name_fmt"], sample_name,
            )
            if gt_path:
                _copy_as_jpg(gt_path, dst)
                entry["gt"] = f"figures/{dataset_safe}_{sample_name_safe}_{panel_index}_gt.jpg"
            else:
                print(f"[WARN] [{cfg['name']}] source GT not found for sample '{sample_name}'")
        else:
            src = checkpoint_image_path(ref_mdir, ref_epoch, "mask", ref_batch)
            if src:
                _save_jpg(load_panel_image(src, "mask"), dst)
                entry["gt"] = f"figures/{dataset_safe}_{sample_name_safe}_{panel_index}_gt.jpg"
        panel_index += 1

        # Per-model prediction + all metrics
        entry["models"] = {}
        for display_name, mdir, source_root, epoch, rows in model_data:
            drive_segdiff_alias = None
            if cfg["name"] == "DRIVE" and display_name in {"SegDiff", "MedSegDiff"} and str(sample_name).isdigit():
                drive_segdiff_alias = f"{int(sample_name):02d}_test"
            sample_row = next(
                (r for r in rows
                 if _normalize_sample_name(r.get("sample_name")) == _normalize_sample_name(sample_name)
                 and (epoch is None or "epoch" not in r or int(r["epoch"]) == epoch)),
                None,
            )
            if sample_row is None and drive_segdiff_alias is not None:
                sample_row = next(
                    (r for r in rows
                     if _normalize_sample_name(r.get("sample_name")) == _normalize_sample_name(drive_segdiff_alias)
                     and (epoch is None or "epoch" not in r or int(r["epoch"]) == epoch)),
                    None,
                )
            if sample_row is None:
                sample_row = next(
                    (r for r in rows
                     if _normalize_sample_name(r.get("sample_name")) == _normalize_sample_name(sample_name)),
                    None,
                )
            if sample_row is None and drive_segdiff_alias is not None:
                sample_row = next(
                    (r for r in rows
                     if _normalize_sample_name(r.get("sample_name")) == _normalize_sample_name(drive_segdiff_alias)),
                    None,
                )
            if sample_row is None and str(sample_name).isdigit():
                sample_batch = int(sample_name)
                sample_row = next(
                    (r for r in rows if r.get("batch") is not None and int(r.get("batch")) == sample_batch),
                    None,
                )
            if sample_row is None and ordered_sample_name is not None:
                sample_row = next(
                    (r for r in rows
                     if _normalize_sample_name(r.get("sample_name")) == _normalize_sample_name(ordered_sample_name)
                     and (epoch is None or "epoch" not in r or int(r["epoch"]) == epoch)),
                    None,
                )
            if sample_row is None and ordered_sample_name is not None:
                sample_row = next(
                    (r for r in rows
                     if _normalize_sample_name(r.get("sample_name")) == _normalize_sample_name(ordered_sample_name)),
                    None,
                )
            model_entry: Dict[str, Any] = {}
            if sample_row is not None:
                src = None
                batch = sample_row.get("batch")
                sample_epoch = sample_row.get("epoch", epoch)
                raw_lookup = _normalize_sample_name(sample_row.get("sample_name") or sample_name)
                sample_lookup_name = re.sub(r'_(?:test|train|val)$', '', raw_lookup, flags=re.IGNORECASE)
                if batch is not None and sample_epoch is not None:
                    src = checkpoint_image_path(mdir, int(sample_epoch), "pred", int(batch))
                if src is None:
                    preferred_fragments = ["_output_ens", "_output"] if display_name == "MedSegDiff" else ["_output"] if display_name == "SegDiff" else None
                    src = _find_fallback_prediction_image(
                        source_root,
                        sample_lookup_name,
                        batch=batch,
                        preferred_fragments=preferred_fragments,
                    )
                if src:
                    model_safe = _figure_panel_token(display_name)
                    dst = os.path.join(figures_dir,
                                       f"{dataset_safe}_{sample_name_safe}_{panel_index}_{model_safe}.jpg")
                    _save_jpg(load_panel_image(src, "pred"), dst)
                    model_entry["pred"] = f"figures/{dataset_safe}_{sample_name_safe}_{panel_index}_{model_safe}.jpg"
                for key in _SAMPLE_METRIC_KEYS:
                    v = _clean_float(sample_row.get(key))
                    if v is not None:
                        model_entry[key] = v
            entry["models"][display_name] = model_entry
            panel_index += 1

        sample_json_path = os.path.join(figures_dir, f"{dataset_safe}_{sample_name_safe}.json")
        _write_json(sample_json_path, entry)
        samples_json[sample_name] = entry

    json_path = os.path.join(output_dir, f"{dataset_safe}.json")
    _write_json(json_path, samples_json)
    print(f"[OUT] {cfg['name']} JSON:    {json_path}")
    print(f"[OUT] {cfg['name']} figures: {figures_dir}/")


# ── Ablation-table qualitative figures ──────────────────────────────────────────
#
# Exports Image/GT/per-row prediction panels for the ablation tables built by
# aggregate_results.py (category-grouped Label Mode x MADF x SegAux, the S-set
# ablation, and the branch-composition ablation) -- the same "Basic figure"
# style as generate_dataset_figure(), but with rows keyed off the experiment
# paths aggregate_results.py already resolved per named ablation slot, in
# summary_experiment_means_ablation_paths.json (so this script doesn't need to
# reimplement that selection logic).

_ABLATION_ROW_GROUPS = [
    ("ablation_table", 3),  # Label Mode x MADF x SegAux
    ("s_variation", 1),     # S-set ablation
    ("imo_variation", 1),   # branch composition ablation
]

_ABLATION_LABEL_MODE_DISPLAY = {
    "dist_inverted": "Dist (Inv)",
    "dist": "Dist",
    "binary": "Binary",
}


def _format_ablation_table_row_key(key: str) -> str:
    """"dist_inverted+madf=True+segaux=True" -> "Dist (Inv) MADF=Y SegAux=Y"."""
    parts = key.split("+")
    label = _ABLATION_LABEL_MODE_DISPLAY.get(parts[0], parts[0])
    flags = []
    for part in parts[1:]:
        flag_name, _, flag_value = part.partition("=")
        flags.append(f"{flag_name.upper()}={'Y' if flag_value == 'True' else 'N'}")
    return label + (" " + " ".join(flags) if flags else "")


def _clean_ablation_set_row_key(key: str) -> str:
    """Strip the \\textbf{...} bold wrapper and brace-escaping LaTeX markup
    aggregate_results.py uses for its S-set / branch-composition row labels,
    e.g. "\\textbf{\\{1,2\\}}" -> "{1,2}"."""
    text = re.sub(r"\\textbf\{((?:\\[{}]|[^{}])*)\}", r"\1", key)
    return text.replace("\\{", "{").replace("\\}", "}")


def _resolve_checkpoint_sample_prediction(
    model_dir: str, sample_id: str,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Locate the predicted-mask PNG and per-sample metrics for `sample_id`
    at a training run's dominant (latest-logged) epoch."""
    sample_csv = os.path.join(model_dir, "test_sample_metrics.csv")
    if not os.path.isfile(sample_csv):
        return None, None
    rows = _parse_sample_metrics_full(sample_csv)
    epoch = _dominant_epoch(rows)
    if epoch is None:
        return None, None
    sample_row = next(
        (r for r in rows if int(r["epoch"]) == epoch
         and _normalize_sample_name(r["sample_name"]) == _normalize_sample_name(sample_id)),
        None,
    )
    if sample_row is None:
        return None, None
    pred_path = checkpoint_image_path(model_dir, epoch, "pred", int(sample_row["batch"]))
    return pred_path, sample_row


def _resolve_main_figure_sample_id(output_dir: str, dataset_safe: str) -> Optional[str]:
    """Read the sample id generate_dataset_figure() already picked for this
    dataset's main comparison figure (output_dir/{dataset}.json), so
    ablation-table figures use the exact same sample instead of an
    independently-computed pick -- the main figure's Dice argmax is
    restricted to samples common to every baseline model, so recomputing it
    from the proposed config's CSV alone can (and did) land on a different,
    higher-Dice-but-baseline-missing sample."""
    json_path = os.path.join(output_dir, f"{dataset_safe}.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    return next(iter(data), None) if data else None


def _resolve_dataset_best_sample_id(model_dir: str) -> Optional[str]:
    """Auto-pick the highest-Dice test sample for a training run -- the same
    per-dataset selection generate_dataset_figure() uses for the main
    comparison figure, so ablation-table figures default to a sample id that
    actually exists for that dataset's own naming scheme (DRIVE uses
    zero-padded numbers, DeepCrack/others use source filenames, etc.)."""
    sample_csv = os.path.join(model_dir, "test_sample_metrics.csv")
    if not os.path.isfile(sample_csv):
        return None
    rows = _parse_sample_metrics_full(sample_csv)
    epoch = _dominant_epoch(rows)
    if epoch is None:
        return None
    epoch_rows = [r for r in rows if int(r["epoch"]) == epoch and r.get("dice") is not None]
    if not epoch_rows:
        return None
    best = max(epoch_rows, key=lambda r: (float(r["dice"]), str(r["sample_name"])))
    return str(best["sample_name"])


def generate_ablation_sample_figures(
    cfg: Dict[str, Any],
    output_dir: str,
    ablation_paths_json: str,
    sample_id: Optional[str] = None,
) -> None:
    """Export Image/GT/per-row prediction panels + JSON for each ablation
    table group available for `cfg`'s dataset, under output_dir/figures/.

    sample_id: if None (default), reuses the sample generate_dataset_figure()
    already picked for this dataset's main comparison figure (so the two
    figures show the same test image); if that hasn't been generated yet,
    falls back to the proposed ablation config's own highest-Dice sample."""
    if not os.path.isfile(ablation_paths_json):
        print(f"[WARN] Ablation paths JSON not found: {ablation_paths_json}")
        return
    with open(ablation_paths_json) as f:
        all_ablation_paths = json.load(f)

    dataset_key = cfg["name"].lower().replace("-", "").replace(" ", "")
    dataset_paths = next(
        (v for k, v in all_ablation_paths.items() if k.lower().replace("-", "").replace(" ", "") == dataset_key),
        None,
    )
    if not dataset_paths:
        print(f"[INFO] [{cfg['name']}] no ablation-table path data in {ablation_paths_json}; skipping.")
        return

    proposed_path = dataset_paths.get("ablation_table", {}).get("proposed", {}).get("path")
    dataset_safe = cfg["name"].lower().replace("-", "_").replace(" ", "_")

    if sample_id is None:
        sample_id = _resolve_main_figure_sample_id(output_dir, dataset_safe)
        if sample_id is not None:
            print(f"[INFO] [{cfg['name']}] ablation: reusing main-figure sample '{sample_id}'.")

    if sample_id is None:
        if proposed_path is None:
            print(f"[WARN] [{cfg['name']}] ablation: no 'proposed' entry to auto-pick a sample from; skipping.")
            return
        proposed_model_dir = _model_dir(proposed_path, cfg["folds"][0])
        sample_id = _resolve_dataset_best_sample_id(proposed_model_dir)
        if sample_id is None:
            print(f"[WARN] [{cfg['name']}] ablation: could not auto-pick a sample from {proposed_model_dir}; skipping.")
            return
        print(f"[INFO] [{cfg['name']}] ablation: main figure not found; auto-selected highest-Dice sample '{sample_id}'.")

    source_data = cfg.get("source_data")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for group_key, num_label_cols in _ABLATION_ROW_GROUPS:
        group = dataset_paths.get(group_key)
        if not group:
            continue
        row_items = [(k, v) for k, v in group.items() if k != "proposed"]
        if not row_items:
            continue

        stem = f"{dataset_safe}_ablation_{group_key}_{sample_id}"
        entry: Dict[str, Any] = {"id": sample_id, "group": group_key}
        panel_index = 1

        dst = os.path.join(figures_dir, f"{stem}_{panel_index}_image.jpg")
        if source_data:
            img_path = _find_source_image(source_data["data_root"], source_data["image_subdir"],
                                           source_data["image_name_fmt"], sample_id)
            if img_path:
                _copy_as_jpg(img_path, dst)
                entry["image"] = f"figures/{os.path.basename(dst)}"
            else:
                print(f"[WARN] [{cfg['name']}] ablation[{group_key}]: source image not found for sample '{sample_id}'")
        panel_index += 1

        dst = os.path.join(figures_dir, f"{stem}_{panel_index}_gt.jpg")
        if source_data:
            gt_path = _find_source_image(source_data["data_root"], source_data["gt_subdir"],
                                          source_data["gt_name_fmt"], sample_id)
            if gt_path:
                _copy_as_jpg(gt_path, dst)
                entry["gt"] = f"figures/{os.path.basename(dst)}"
            else:
                print(f"[WARN] [{cfg['name']}] ablation[{group_key}]: source GT not found for sample '{sample_id}'")
        panel_index += 1

        entry["rows"] = {}
        for raw_key, info in row_items:
            display_label = (
                _format_ablation_table_row_key(raw_key) if num_label_cols == 3
                else _clean_ablation_set_row_key(raw_key)
            )
            experiment_root = info.get("path")
            row_entry: Dict[str, Any] = {
                "label": display_label,
                "highlighted": bool(proposed_path) and experiment_root == proposed_path,
            }
            model_dir = _model_dir(experiment_root, cfg["folds"][0]) if experiment_root else None
            if model_dir is not None:
                pred_path, sample_row = _resolve_checkpoint_sample_prediction(model_dir, sample_id)
                if pred_path:
                    row_safe = re.sub(r"[^a-z0-9]+", "_", display_label.lower()).strip("_")
                    dst = os.path.join(figures_dir, f"{stem}_{panel_index}_{row_safe}.jpg")
                    _save_jpg(load_panel_image(pred_path, "pred"), dst)
                    row_entry["pred"] = f"figures/{os.path.basename(dst)}"
                else:
                    print(f"[WARN] [{cfg['name']}] ablation[{group_key}]: no prediction for '{display_label}' "
                          f"(sample '{sample_id}')")
                if sample_row is not None:
                    for key in _SAMPLE_METRIC_KEYS:
                        if key in sample_row and sample_row[key] is not None:
                            row_entry[key] = sample_row[key]
            entry["rows"][raw_key] = row_entry
            panel_index += 1

        json_path = os.path.join(figures_dir, f"{stem}.json")
        _write_json(json_path, entry)
        print(f"[OUT] {cfg['name']} ablation[{group_key}] JSON: {json_path}")

    print(f"[OUT] {cfg['name']} ablation figures: {figures_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper qualitative figures.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--datasets", nargs="+", metavar="NAME", default=DEFAULT_DATASETS,
        help="Dataset names to include (default: all).",
    )
    parser.add_argument(
        "--list", metavar="DATASET",
        help="Print sorted experiment list for a dataset and exit (for filling experiment_keys).",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="output",
        help=(
            "Comma-separated list of DconnNet result roots. Each dataset is resolved under "
            "these roots (e.g., output,output_dropone_svariation)."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Also export per-row qualitative panels for the ablation tables "
            "(category-grouped Label Mode x MADF x SegAux, S-set, branch "
            "composition) built by aggregate_results.py, for datasets with "
            "entries in --ablation-paths-json."
        ),
    )
    parser.add_argument(
        "--ablation-paths-json",
        default="output/summary/summary_experiment_means_ablation_paths.json",
        help="Path to the ablation experiment-paths JSON written by aggregate_results.py.",
    )
    parser.add_argument(
        "--ablation-sample",
        default=None,
        help=(
            "Sample id to render for the ablation-table row panels. "
            "Default: auto-pick the proposed config's own highest-Dice "
            "sample per dataset (sample ids aren't shared across datasets)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_roots = parse_input_roots(args.input_root)

    if args.list:
        matches = [c for c in DATASET_CONFIGS if c["name"] == args.list]
        if not matches:
            known = [c["name"] for c in DATASET_CONFIGS]
            print(f"[ERROR] Unknown dataset '{args.list}'. Known: {known}")
            sys.exit(1)
        list_dataset_experiments(matches[0], input_roots=input_roots)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    configs = DATASET_CONFIGS
    if args.datasets:
        configs = [c for c in configs if c["name"] in args.datasets]
        if not configs:
            known = [c["name"] for c in DATASET_CONFIGS]
            print(f"[ERROR] No matching dataset configs. Known: {known}")
            sys.exit(1)

    print("[INFO] ── Figures ──")
    for cfg in configs:
        try:
            generate_dataset_figure(cfg, args.output_dir, input_roots=input_roots)
        except Exception as e:
            print(f"[WARN] [{cfg['name']}] figure failed: {e}")

    if args.ablation:
        print("[INFO] ── Ablation-table figures ──")
        for cfg in configs:
            try:
                generate_ablation_sample_figures(
                    cfg, args.output_dir, args.ablation_paths_json, sample_id=args.ablation_sample,
                )
            except Exception as e:
                print(f"[WARN] [{cfg['name']}] ablation figure failed: {e}")


if __name__ == "__main__":
    main()
