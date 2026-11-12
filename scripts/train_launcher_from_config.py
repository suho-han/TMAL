#!/usr/bin/env python3
"""Config-first train launcher.

Usage:
  scripts/train_launcher.sh --config <yaml> [--device N] [--batch-size N] [--dry_run] [--smoke] [--smoke_limit N]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CANONICAL_8_OFFSETS = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]

OUTER_8_OFFSETS = [
    (-2, -2),
    (-2, 0),
    (-2, 2),
    (0, -2),
    (0, 2),
    (2, -2),
    (2, 0),
    (2, 2),
]

SCALE3_8_OFFSETS = [
    (-3, -3),
    (-3, 0),
    (-3, 3),
    (0, -3),
    (0, 3),
    (3, -3),
    (3, 0),
    (3, 3),
]

SCALE4_8_OFFSETS = [
    (-4, -4),
    (-4, 0),
    (-4, 4),
    (0, -4),
    (0, 4),
    (4, -4),
    (4, 0),
    (4, 4),
]

CONNECTIVITY_LAYOUTS = {
    "standard8": {
        "name": "standard8",
        "channel_count": 8,
        "kernel_size": 3,
        "include_center": False,
        "offsets": CANONICAL_8_OFFSETS,
    },
    "full24": {
        "name": "full24",
        "channel_count": 24,
        "kernel_size": 5,
        "include_center": False,
        "offsets": [],  # Not needed for launcher
    },
    "out8": {
        "name": "out8",
        "channel_count": 8,
        "kernel_size": 5,
        "include_center": False,
        "offsets": OUTER_8_OFFSETS,
    },
    "scale3_8": {
        "name": "scale3_8",
        "channel_count": 8,
        "kernel_size": 7,
        "include_center": False,
        "offsets": SCALE3_8_OFFSETS,
    },
    "scale4_8": {
        "name": "scale4_8",
        "channel_count": 8,
        "kernel_size": 9,
        "include_center": False,
        "offsets": SCALE4_8_OFFSETS,
    },
}


def default_connectivity_layout_name(conn_num: int) -> str:
    if conn_num == 8:
        return "standard8"
    if conn_num == 24:
        return "full24"
    raise ValueError(f"Unsupported conn_num {conn_num}, only 8 and 24 are supported")


def normalize_conn_layout(conn_num: int, conn_layout: str | None = None) -> str:
    layout_name = (
        default_connectivity_layout_name(conn_num)
        if conn_layout is None
        else str(conn_layout)
    )
    if layout_name not in CONNECTIVITY_LAYOUTS:
        raise ValueError(
            f"Unsupported conn_layout {layout_name}, "
            f"supported: {sorted(CONNECTIVITY_LAYOUTS)}"
        )

    if conn_num == 8 and layout_name not in {"standard8", "mid8", "out8", "scale3_8", "scale4_8"}:
        raise ValueError("conn_num=8 supports only conn_layout in {'standard8', 'mid8', 'out8', 'scale3_8', 'scale4_8'}")
    if conn_num == 24 and layout_name != "full24":
        raise ValueError("conn_num=24 supports only conn_layout='full24'")
    return layout_name


def is_default_connectivity_layout(conn_num: int, conn_layout: str) -> bool:
    return (
        normalize_conn_layout(conn_num, conn_layout)
        == default_connectivity_layout_name(conn_num)
    )


is_default_conn_layout = is_default_connectivity_layout
default_conn_layout_name = default_connectivity_layout_name

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "PyYAML is required for config launcher. Install dependencies to provide `yaml` module."
    ) from exc

ALLOWED_DATASETS = {
    "chase",
    "cremi",
    "deepcrack",
    "deepglobe",
    "drive",
    "fives",
    "isic2018",
    "octa",
    "octa500",
    "xacv",
}
ALLOWED_EXTERNAL_DATASETS = {"octa", "octa-6M", "octa-3M", "octa500", "octa500-6M", "octa500-3M", "chase", "xacv"}
ALLOWED_MODES = {"single", "multi"}
ALLOWED_OCTA_VARIANTS = {"3M", "6M"}
ALLOWED_RETOUCH_DEVICES = {"Cirrus", "Spectrailis", "Topcon"}
ALLOWED_MONITOR_METRICS = {"val_dice", "val_loss"}
ALLOWED_LABEL_MODES = {"binary", "dist", "dist_inverted"}
ALLOWED_CONN_FUSIONS = {"none", "gate", "scaled_sum", "conv_residual", "decoder_guided", "dg", "dg_direct", "dsf", "dg_dsf"}
ALLOWED_FINAL_DECISIONS = {"max", "mean", "weighted_mean"}
LEGACY_CONN_FUSION_ALIASES = {
    "decoder_guided": "dg",
}
DATASET_PRESETS = {
    "chase": {
        "dataset": "chase",
        "data_root": "data/chase",
        "resize": (960, 960),
        "batch_size": 4,
    },
    "cremi": {
        "dataset": "cremi",
        "data_root": "data/CREMI",
        "resize": (256, 256),
        "batch_size": 4,
    },
    "drive": {
        "dataset": "drive",
        "data_root": "data/DRIVE",
        "resize": (512, 512),
        "batch_size": 4,
    },
    "deepcrack": {
        "dataset": "deepcrack",
        "data_root": "data/DeepCrack",
        "resize": (512, 512),
        "batch_size": 4,
        "lr": 1e-4,
        "lr_update": "CosineAnnealingLR",
        "optimizer": "adamw",
        "weight_decay": 1e-4,
    },
    "deepglobe": {
        "dataset": "deepglobe",
        "data_root": "data/DeepGlobe/splitted",
        "resize": (1024, 1024),
        "batch_size": 2,
    },
    "isic2018": {
        "dataset": "isic",
        "data_root": "data/ISIC2018",
        "resize": (224, 320),
        "batch_size": 64,
    },
    "xacv": {
        "dataset": "xacv",
        "data_root": "data/XACV",
        "resize": (512, 512),
        "batch_size": 16,
    },
    "fives": {
        "dataset": "fives",
        "data_root": "data/FIVES",
        "resize": (512, 512),
        "batch_size": 8,
    },
    "octa": {
        "dataset": "octa500-6M",
        "data_root": "data/OCTA500_6M",
        "resize": (512, 512),
        "batch_size": 16,
        "num_class": 1,
        "lr": 0.0038,
        "lr_update": "poly",
        "use_sdl": False,
    },
    "octa500": {
        "dataset": "octa500-6M",
        "data_root": "data/OCTA500_6M",
        "resize": (512, 512),
        "batch_size": 16,
        "num_class": 1,
        "lr": 0.0038,
        "lr_update": "poly",
        "use_sdl": False,
    },
}


def normalize_conn_fusion(value: Any) -> str:
    conn_fusion = str(value if value is not None else "none")
    conn_fusion = LEGACY_CONN_FUSION_ALIASES.get(conn_fusion, conn_fusion)
    if conn_fusion not in ALLOWED_CONN_FUSIONS:
        raise ValueError(
            f"Unsupported conn_fusion: {conn_fusion} "
            f"(supported: {sorted(ALLOWED_CONN_FUSIONS)})"
        )
    return conn_fusion


def normalize_fusion_branches(value: Any, default: str = "none") -> str:
    if value is None:
        return default
    if isinstance(value, list):
        if len(value) == 1:
            return str(value[0])
        return "".join(str(v) for v in value)
    return str(value)


def normalize_final_decision(value: Any) -> str:
    final_decision = str(value if value is not None else "max")
    if final_decision not in ALLOWED_FINAL_DECISIONS:
        raise ValueError(
            f"Unsupported final_decision: {final_decision} "
            f"(supported: {sorted(ALLOWED_FINAL_DECISIONS)})"
        )
    return final_decision


def validate_final_decision_compatibility(
    final_decision: str,
    conn_layout: str,
    fusion_branches: str = "none",
) -> None:
    if final_decision == "max":
        return
    if conn_layout != "standard8":
        raise ValueError(
            "final_decision other than max currently supports only "
            "conn_layout=standard8"
        )
    # Non-standard auxiliary branches keep their native max reduction.
    # The selected decision applies to the standard8 fused/main path.


def normalize_label_mode(value: Any) -> str:
    label_mode = str(value if value is not None else "binary")
    if label_mode not in ALLOWED_LABEL_MODES:
        raise ValueError(
            f"Unsupported label_mode: {label_mode} "
            f"(supported: {sorted(ALLOWED_LABEL_MODES)})"
        )
    return label_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-first dataset launcher."
    )
    parser.add_argument("--config", required=True, help="Path to launcher YAML config")
    parser.add_argument("--device", type=int, default=None, help="Optional device override")
    parser.add_argument("--dry_run", action="store_true", help="Print generated commands only")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a minimal smoke run across all generated run(s) by default "
            "(5 epochs, batch_size=1, target_fold=1, output_dir=output_smoke)."
        ),
    )
    parser.add_argument(
        "--smoke_limit",
        type=int,
        default=None,
        help="When --smoke is set, optionally limit to first N generated run(s).",
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Run generated train.py command(s) in test-only mode",
    )
    parser.add_argument(
        "--pretrained",
        nargs="?",
        const="__AUTO__",
        default=None,
        help=(
            "Optional pretrained path for test-only runs. "
            "If omitted after flag, auto-resolve from output_dir/dataset/experiment_name."
        ),
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None, help="Optional batch size override")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        f = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {path}")
    with f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def apply_smoke_overrides(config: dict[str, Any], mode: str) -> None:
    if mode == "single":
        block_key = "single"
    elif mode == "multi":
        block_key = "multi"
    else:
        raise ValueError(f"Unsupported mode for smoke overrides: {mode}")

    block_cfg = config.get(block_key)
    if block_cfg is None:
        block_cfg = {}
        config[block_key] = block_cfg
    if not isinstance(block_cfg, dict):
        raise ValueError(f"{block_key} block must be a mapping for smoke overrides")

    block_cfg["epochs"] = 5
    block_cfg["save_best_only"] = True
    # In train.py, early stopping is disabled when patience <= 0.
    block_cfg["early_stopping_patience"] = 0


def ensure_isic_dataset_ready(repo_root: Path) -> None:
    image_dir = repo_root / "data" / "ISIC2018" / "image"
    label_dir = repo_root / "data" / "ISIC2018" / "label"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(
            "Missing prepared ISIC2018 npy dataset under data/ISIC2018/{image,label}. "
            "Run scripts/prepare_isic2018_npy.py first."
        )


def as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return list(value)
    return [value]


def get_with_alias(cfg: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def normalize_datasets_for_mode(dataset_value: Any, mode: str) -> list[str]:
    if mode == "single":
        if isinstance(dataset_value, list):
            raise ValueError("single mode requires `dataset` to be a string, not a list")
        if not isinstance(dataset_value, str):
            raise ValueError("single mode requires `dataset` to be a string")
        return [dataset_value]

    if isinstance(dataset_value, str):
        datasets = [dataset_value]
    elif isinstance(dataset_value, list):
        datasets = list(dataset_value)
    else:
        raise ValueError("multi mode requires `dataset` to be a string or list of strings")

    if not datasets:
        raise ValueError("multi mode requires at least one dataset")
    if any(not isinstance(dataset, str) for dataset in datasets):
        raise ValueError("multi mode requires all `dataset` entries to be strings")
    return datasets


def format_dataset_summary(datasets: list[str]) -> str:
    if len(datasets) == 1:
        return datasets[0]
    return "[" + ",".join(datasets) + "]"


def default_single_epochs(dataset: str, conn_num: int, label_mode: str) -> int:
    if dataset.startswith("retouch"):
        return 50
    if dataset == "isic2018":
        return 500
    if dataset == "deepcrack":
        return 100
    if label_mode in {"dist", "dist_inverted"}:
        return 260
    return 390 if conn_num == 24 else 130


def resolve_preset(
    dataset: str,
    octa_variant: str | None = None,
    retouch_device: str | None = None,
    retouch_data_root: str | None = None,
) -> dict[str, Any]:
    if dataset not in {"octa", "octa500", "retouch"}:
        return DATASET_PRESETS[dataset]
    if octa_variant not in ALLOWED_OCTA_VARIANTS:
        if dataset in {"octa", "octa500"}:
            raise ValueError(f"Unsupported octa variant: {octa_variant} (supported: 3M, 6M)")

    if dataset in {"octa", "octa500"}:
        batch_size = 8 if octa_variant == "3M" else 16
        return {
            "dataset": f"octa500-{octa_variant}",
            "data_root": f"data/OCTA500_{octa_variant}",
            "resize": (512, 512),
            "batch_size": batch_size,
            "num_class": 1,
            "lr": 0.0038,
            "lr_update": "poly",
            "use_sdl": False,
        }

    if retouch_device not in ALLOWED_RETOUCH_DEVICES:
        raise ValueError(
            f"Unsupported retouch device: {retouch_device} "
            f"(supported: {sorted(ALLOWED_RETOUCH_DEVICES)})"
        )
    data_root = retouch_data_root or "data/retouch"
    lr = 0.0008 if retouch_device == "Topcon" else 0.00085
    return {
        "dataset": f"retouch-{retouch_device}",
        "data_root": data_root,
        "resize": (256, 256),
        "batch_size": 8,
        "num_class": 4,
        "lr": lr,
        "lr_update": "poly",
        "use_sdl": False,
    }


def resolve_external_preset(external_data: str) -> dict[str, Any]:
    if external_data in ("octa500", "octa500-6M", "octa", "octa-6M"):
        return resolve_preset("octa500", octa_variant="6M")
    elif external_data in ("octa500-3M", "octa-3M"):
        return resolve_preset("octa500", octa_variant="3M")
    elif external_data == "chase":
        preset = dict(resolve_preset("chase"))
        preset["resize"] = (512, 512)  # match DRIVE training resolution
        return preset
    elif external_data == "xacv":
        return {
            "dataset": "xacv",
            "data_root": "data/XACV",
            "resize": (512, 512),
            "batch_size": 4,
        }
    else:
        raise ValueError(
            f"Unsupported external_data: {external_data} "
            f"(supported: {sorted(ALLOWED_EXTERNAL_DATASETS)})"
        )


def parse_retouch_devices(value: Any, default: list[str]) -> list[str]:
    devices = [str(v) for v in as_list(value, default)]
    unsupported = sorted({device for device in devices if device not in ALLOWED_RETOUCH_DEVICES})
    if unsupported:
        raise ValueError(
            f"Unsupported retouch device(s): {unsupported} "
            f"(supported: {sorted(ALLOWED_RETOUCH_DEVICES)})"
        )
    return devices


def parse_training_control_overrides(block_cfg: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if "monitor_metric" in block_cfg:
        monitor_metric = str(block_cfg["monitor_metric"])
        if monitor_metric not in ALLOWED_MONITOR_METRICS:
            raise ValueError(
                f"Unsupported monitor_metric: {monitor_metric} "
                f"(supported: {sorted(ALLOWED_MONITOR_METRICS)})"
            )
        overrides["monitor_metric"] = monitor_metric

    if "early_stopping_patience" in block_cfg:
        patience = int(block_cfg["early_stopping_patience"])
        if patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        overrides["early_stopping_patience"] = patience

    if "early_stopping_min_delta" in block_cfg:
        min_delta = float(block_cfg["early_stopping_min_delta"])
        if min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        overrides["early_stopping_min_delta"] = min_delta

    if "early_stopping_tie_eps" in block_cfg:
        tie_eps = float(block_cfg["early_stopping_tie_eps"])
        if tie_eps < 0:
            raise ValueError("early_stopping_tie_eps must be >= 0")
        overrides["early_stopping_tie_eps"] = tie_eps

    if "early_stopping_stop_interval" in block_cfg:
        stop_interval = int(block_cfg["early_stopping_stop_interval"])
        if stop_interval < 1:
            raise ValueError("early_stopping_stop_interval must be >= 1")
        overrides["early_stopping_stop_interval"] = stop_interval

    if "tie_break_with_loss" in block_cfg:
        tie_break_with_loss = block_cfg["tie_break_with_loss"]
        if not isinstance(tie_break_with_loss, bool):
            raise ValueError("tie_break_with_loss must be boolean")
        overrides["tie_break_with_loss"] = tie_break_with_loss

    if "save_best_only" in block_cfg:
        save_best_only = block_cfg["save_best_only"]
        if not isinstance(save_best_only, bool):
            raise ValueError("save_best_only must be boolean")
        overrides["save_best_only"] = save_best_only

    return overrides


def parse_bool_config(config: dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in config:
        return default
    value = config[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _normalize_experiment_datasets(exp_cfg: dict[str, Any], default_datasets: list[str]) -> list[str]:
    exp_datasets_raw = exp_cfg.get("datasets", exp_cfg.get("dataset"))
    if exp_datasets_raw is None:
        return list(default_datasets)
    datasets = normalize_datasets_for_mode(exp_datasets_raw, mode="multi")
    unknown = sorted(set(datasets) - set(default_datasets))
    if unknown:
        raise ValueError(
            "multi.experiments specifies dataset(s) not present in top-level `dataset`: "
            + ", ".join(unknown)
        )
    return datasets


REMOVED_DECODER_FUSION_KEYS = ("decoder_fusion", "decoder_fusions", "lambda_vote_aux")
REMOVED_FUSION_LOSS_PROFILE_KEYS = (
    "fusion_loss_profile",
    "fusion_loss_profiles",
    "fusion_lambda_inner",
    "fusion_lambda_inners",
    "fusion_lambda_outer",
    "fusion_lambda_outers",
)


def reject_removed_decoder_fusion_keys(config: dict[str, Any], context: str) -> None:
    for key in REMOVED_DECODER_FUSION_KEYS:
        if key in config:
            raise ValueError(
                f"{context} uses removed key `{key}`. "
                "Explicit decoder_fusion support has been removed."
            )
    for key in REMOVED_FUSION_LOSS_PROFILE_KEYS:
        if key in config:
            raise ValueError(
                f"{context} uses removed key `{key}`. "
                "fusion_loss_profile A/B/C and fusion_lambda_inner/outer have been removed; "
                "supervision is now driven by fused loss + conn_aux_c*_weight only."
            )


def build_experiment_output_name(
    conn_num: int,
    label_mode: str,
    dist_aux_loss: str,
    conn_layout: str | None = None,
    conn_fusion: str = "none",
    fusion_residual_scale: float = 0.2,
    use_seg_aux: bool = False,
    seg_aux_weight: float | None = None,
    experiment_name_suffix: str | None = None,
    fusion_branches: str = "none",
    final_decision: str = "max",
) -> str:
    layout_name = normalize_conn_layout(conn_num, conn_layout)
    final_decision = normalize_final_decision(final_decision)
    layout_suffix = "" if is_default_conn_layout(conn_num, layout_name) else f"_{layout_name}"
    conn_fusion_name = normalize_conn_fusion(conn_fusion)
    validate_final_decision_compatibility(
        final_decision,
        layout_name,
        fusion_branches if conn_fusion_name != "none" else "none",
    )
    if conn_fusion_name == "none":
        if label_mode == "binary":
            base_name = f"binary_{conn_num}{layout_suffix}_bce"
        else:
            base_name = f"{label_mode}_{conn_num}{layout_suffix}_{dist_aux_loss}"
    else:
        fusion_tag = conn_fusion_name
        if conn_fusion_name == "scaled_sum":
            scale_str = f"{float(fusion_residual_scale):.6f}".rstrip("0").rstrip(".")
            fusion_tag = f"{fusion_tag}_rs{scale_str}"
        fb = str(fusion_branches or "none").strip().lower()
        if fb and fb != "none":
            fusion_tag = f"{fusion_tag}_{fb}"
        if label_mode == "binary":
            base_name = f"binary_{fusion_tag}_{conn_num}{layout_suffix}_bce"
        else:
            base_name = f"{label_mode}_{fusion_tag}_{conn_num}{layout_suffix}_{dist_aux_loss}"

    if use_seg_aux:
        if seg_aux_weight is not None:
            base_name = f"{base_name}_segaux_w{seg_aux_weight}"
        else:
            base_name = f"{base_name}_segaux"

    if final_decision != "max":
        base_name = f"{base_name}_fd_{final_decision}"

    suffix = str(experiment_name_suffix or "").strip()
    if suffix:
        base_name = f"{base_name}_{suffix}"

    return base_name


def resolve_pretrained_path(
    pretrained_spec: str | None,
    output_dir: str,
    dataset: str,
    conn_num: int,
    label_mode: str,
    dist_aux_loss: str,
    conn_layout: str | None = None,
    conn_fusion: str = "none",
    fusion_residual_scale: float = 0.2,
    use_seg_aux: bool = False,
    seg_aux_weight: float | None = None,
    experiment_name_suffix: str | None = None,
    fusion_branches: str = "none",
    final_decision: str = "max",
) -> str:
    exp_name = build_experiment_output_name(
        conn_num=conn_num,
        label_mode=label_mode,
        dist_aux_loss=dist_aux_loss,
        conn_layout=conn_layout,
        conn_fusion=conn_fusion,
        fusion_residual_scale=fusion_residual_scale,
        use_seg_aux=use_seg_aux,
        seg_aux_weight=seg_aux_weight,
        experiment_name_suffix=experiment_name_suffix,
        fusion_branches=fusion_branches,
        final_decision=final_decision,
    )
    models_dir = Path(output_dir) / dataset / exp_name / "models"

    if pretrained_spec:
        try:
            resolved = pretrained_spec.format(
                dataset=dataset,
                conn_num=conn_num,
                label_mode=label_mode,
                dist_aux_loss=dist_aux_loss,
                conn_layout=normalize_conn_layout(conn_num, conn_layout),
                conn_fusion=normalize_conn_fusion(conn_fusion),
                final_decision=normalize_final_decision(final_decision),
                use_seg_aux=use_seg_aux,
                seg_aux_weight=seg_aux_weight,
                experiment_name=exp_name,
            )
        except KeyError as exc:
            raise ValueError(
                f"Unsupported pretrained format key: {exc} "
                "(supported: dataset, conn_num, label_mode, dist_aux_loss, conn_layout, "
                "conn_fusion, final_decision, experiment_name)"
            ) from exc
        resolved_path = Path(resolved)
        if resolved_path.is_absolute():
            return str(resolved_path)

        normalized = resolved.replace("\\", "/")
        if normalized.startswith("output/") or normalized.startswith("./") or normalized.startswith("../"):
            return resolved
        if normalized.startswith("models/"):
            normalized = normalized[len("models/"):]
        return str(models_dir / normalized)

    # Prefer 500_model.pth; fall back to the largest {N}_model.pth available.
    preferred = models_dir / "500_model.pth"
    if preferred.exists():
        return str(preferred)
    epoch_models = sorted(
        models_dir.glob("[0-9]*_model.pth"),
        key=lambda p: int(p.stem.split("_")[0]),
    )
    if epoch_models:
        return str(epoch_models[-1])
    return str(preferred)  # non-existent path; runtime check will report it clearly


def build_train_cmd(
    pybin: str,
    repo_root: Path,
    preset: dict[str, Any],
    conn_num: int,
    label_mode: str,
    dist_aux_loss: str,
    conn_layout: str,
    dist_sf_l1_gamma: float,
    device: int,
    epochs: int,
    folds: int,
    target_fold: int | None,
    training_ctrl_overrides: dict[str, Any] | None = None,
    model: str = "dconnnet",
    output_dir: str | None = None,
    pretrained: str | None = None,
    test_only: bool = False,
    conn_fusion: str = "none",
    fusion_branches: str = "io",
    final_decision: str = "max",
    fusion_lambda_middle: float = 0.05,
    fusion_lambda_fused: float = 0.3,
    fusion_residual_scale: float = 0.2,
    use_seg_aux: bool = False,
    seg_aux_weight: float | None = None,
    fusion_gate_reg_weight: float = 0.01,
    conn_aux_c3_weight: float = 0.3,
    conn_aux_c4_weight: float = 0.2,
    conn_aux_c5_weight: float = 0.2,
    conn_aux_c6_weight: float = 0.3,
    conn_aux_c7_weight: float = 0.2,
    legacy_aux_weight: float = 0.0,
    debug_loss_breakdown: bool = False,
    simple_loss: bool = False,
    experiment_name_suffix: str | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    num_channels: int | None = None,
    diff_timesteps: int = 100,
    diff_objective: str = 'pred_noise',
    beta_schedule: str = 'linear',
    image_size: int = 128,
    save_per_epochs: int | None = None,
    external_data: str | None = None,
) -> list[str]:
    resize_h, resize_w = preset["resize"]
    cli_dist_aux_loss = "smooth_l1" if label_mode == "binary" and str(dist_aux_loss) == "bce" else dist_aux_loss
    cmd = [
        pybin,
        str(repo_root / "src" / "engine" / ("train_ablation.py" if (bool(simple_loss) or model in {"segdiff", "medsegdiff"}) else "train.py")),
        "--dataset",
        str(preset["dataset"]),
        "--data_root",
        str(preset["data_root"]),
        "--resize",
        str(resize_h),
        str(resize_w),
        "--num-class",
        str(preset.get("num_class", 1)),
        "--batch-size",
        str(preset["batch_size"]),
        "--epochs",
        str(epochs),
        "--lr",
        str(preset.get("lr", 0.0038)),
        "--lr-update",
        str(preset.get("lr_update", "poly")),
        "--optimizer",
        str(preset.get("optimizer", "adam")),
        "--weight_decay",
        str(preset.get("weight_decay", 0.0)),
        "--folds",
        str(folds),
        "--conn_num",
        str(conn_num),
        "--conn_layout",
        str(conn_layout),
        "--label_mode",
        str(label_mode),
        "--dist_aux_loss",
        str(cli_dist_aux_loss),
        "--dist_sf_l1_gamma",
        str(dist_sf_l1_gamma),
        "--final_decision",
        str(normalize_final_decision(final_decision)),
        "--conn_fusion",
        str(conn_fusion),
        "--fusion_branches",
        str(fusion_branches),
        "--fusion_lambda_middle",
        str(fusion_lambda_middle),
        "--fusion_lambda_fused",
        str(fusion_lambda_fused),
        "--fusion_residual_scale",
        str(fusion_residual_scale),
        "--device",
        str(device),
        "--model",
        str(model),
    ]
    if num_channels is not None:
        cmd += ["--num_channels", str(num_channels)]
    if model in {"segdiff", "medsegdiff"}:
        cmd += [
            "--diff_timesteps", str(diff_timesteps),
            "--diff_objective", diff_objective,
            "--beta_schedule", beta_schedule,
        ]
    if model == "medsegdiff":
        cmd += ["--image_size", str(image_size)]
    if save_per_epochs is not None:
        cmd.extend(["--save-per-epochs", str(save_per_epochs)])
    if use_seg_aux:
        cmd.append("--use_seg_aux")
        if seg_aux_weight is not None:
            cmd.extend(["--seg_aux_weight", str(seg_aux_weight)])
    if conn_fusion in {"dg", "dg_direct", "dg_dsf"}:
        cmd.extend(["--fusion_gate_reg_weight", str(fusion_gate_reg_weight)])
    if conn_fusion in {"dg", "dg_direct", "dsf", "dg_dsf"}:
        cmd.extend(["--conn_aux_c3_weight", str(conn_aux_c3_weight)])
        if 'm' in str(fusion_branches):
            cmd.extend(["--conn_aux_c4_weight", str(conn_aux_c4_weight)])
        cmd.extend(["--conn_aux_c5_weight", str(conn_aux_c5_weight)])
        if 'p' in str(fusion_branches):
            cmd.extend(["--conn_aux_c6_weight", str(conn_aux_c6_weight)])
        if 'f' in str(fusion_branches):
            cmd.extend(["--conn_aux_c7_weight", str(conn_aux_c7_weight)])
    if conn_fusion != "none" and float(legacy_aux_weight) > 0.0:
        cmd.extend(["--legacy_aux_weight", str(float(legacy_aux_weight))])
    if bool(debug_loss_breakdown):
        cmd.append("--debug_loss_breakdown")
    if bool(simple_loss):
        cmd.append("--simple_loss")
    if target_fold is not None:
        cmd.extend(["--target_fold", str(target_fold)])
    if max_train_batches is not None:
        cmd.extend(["--max_train_batches", str(max_train_batches)])
    if max_eval_batches is not None:
        cmd.extend(["--max_eval_batches", str(max_eval_batches)])
    if output_dir:
        cmd.extend(["--output_dir", str(output_dir)])
    if experiment_name_suffix:
        cmd.extend(["--experiment_name_suffix", str(experiment_name_suffix)])
    if pretrained:
        cmd.extend(["--pretrained", str(pretrained)])
    if test_only:
        cmd.append("--test_only")
    if external_data:
        cmd.extend(["--external_data", str(external_data)])
    if training_ctrl_overrides:
        if "monitor_metric" in training_ctrl_overrides:
            cmd.extend(["--monitor_metric", str(training_ctrl_overrides["monitor_metric"])])
        if "early_stopping_patience" in training_ctrl_overrides:
            cmd.extend(
                ["--early_stopping_patience", str(training_ctrl_overrides["early_stopping_patience"])]
            )
        if "early_stopping_min_delta" in training_ctrl_overrides:
            cmd.extend(
                ["--early_stopping_min_delta", str(training_ctrl_overrides["early_stopping_min_delta"])]
            )
        if "early_stopping_tie_eps" in training_ctrl_overrides:
            cmd.extend(
                ["--early_stopping_tie_eps", str(training_ctrl_overrides["early_stopping_tie_eps"])]
            )
        if "early_stopping_stop_interval" in training_ctrl_overrides:
            cmd.extend(
                [
                    "--early_stopping_stop_interval",
                    str(training_ctrl_overrides["early_stopping_stop_interval"]),
                ]
            )
        if "tie_break_with_loss" in training_ctrl_overrides:
            cmd.append(
                "--tie_break_with_loss"
                if bool(training_ctrl_overrides["tie_break_with_loss"])
                else "--no_tie_break_with_loss"
            )
        if "save_best_only" in training_ctrl_overrides:
            cmd.append(
                "--save_best_only"
                if bool(training_ctrl_overrides["save_best_only"])
                else "--no_save_best_only"
            )
    return cmd


def read_run_final_metrics(run: dict[str, Any]) -> dict[str, str] | None:
    """Return the first 'final' row from final_results.csv for a completed run, or None."""
    exp_dir = get_run_output_dir(run)
    folds = int(run.get("folds", 1))
    target_fold = run.get("target_fold")
    fold_ids = [int(target_fold)] if target_fold is not None else list(range(1, folds + 1))

    candidates: list[Path] = []
    for fold in fold_ids:
        candidates.append(exp_dir / f"final_results_{fold}.csv")
        if int(fold) == 1:
            candidates.append(exp_dir / "final_results.csv")

    for csv_path in candidates:
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("result_type", "").lower() in ("final", ""):
                        return row
        except Exception:
            pass
    return None


def format_run_metrics_line(run: dict[str, Any], metrics: dict[str, str] | None) -> str:
    """Format a multi-line metric block for one run."""
    exp_name = get_run_experiment_name(run)
    if metrics is None:
        return f"{exp_name}\n  (no results)"
    try:
        lines = [exp_name]
        for key, label in [("dice", "Dice"), ("jac", "IoU"), ("cldice", "clDice"), ("hd95", "HD95")]:
            val = metrics.get(key, "")
            if val and val.lower() != "nan":
                lines.append(f"  {label}: {float(val):.4f}")
        return "\n".join(lines)
    except (ValueError, TypeError):
        return f"{exp_name}\n  (parse error)"


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY_RUN] {' '.join(shlex.quote(p) for p in cmd)}")
        return
    subprocess.run(cmd, check=True)


def get_run_experiment_name(run: dict[str, Any]) -> str:
    return build_experiment_output_name(
        conn_num=int(run["conn_num"]),
        label_mode=str(run["label_mode"]),
        dist_aux_loss=str(run["dist_aux_loss"]),
        conn_layout=str(run["conn_layout"]),
        conn_fusion=str(run.get("conn_fusion", "none")),
        fusion_residual_scale=float(run.get("fusion_residual_scale", 0.2)),
        use_seg_aux=bool(run.get("use_seg_aux", False)),
        seg_aux_weight=(
            None
            if run.get("seg_aux_weight") is None
            else float(run.get("seg_aux_weight"))
        ),
        experiment_name_suffix=str(run.get("experiment_name_suffix", "") or ""),
        fusion_branches=normalize_fusion_branches(run.get("fusion_branches", "none")),
        final_decision=normalize_final_decision(run.get("final_decision", "max")),
    )


def get_run_output_root(run: dict[str, Any]) -> Path:
    return Path(str(run.get("output_dir", "output")))


def get_run_output_dir(run: dict[str, Any]) -> Path:
    preset = run["preset"]
    dataset = str(preset["dataset"])
    model = str(run.get("model", "dconnnet"))
    root = get_run_output_root(run)
    external_data = run.get("external_data")
    if model != "dconnnet":
        return root / model / dataset
    if external_data:
        return root / f"drive_to_{external_data}" / get_run_experiment_name(run)
    return root / dataset / get_run_experiment_name(run)


def get_run_completion_status(run: dict[str, Any]) -> tuple[bool, str | None]:
    exp_dir = get_run_output_dir(run)

    target_fold = run.get("target_fold")
    fold_ids = [int(target_fold)] if target_fold is not None else list(range(1, int(run["folds"]) + 1))

    for fold in fold_ids:
        final_csv = exp_dir / f"final_results_{fold}.csv"
        if final_csv.exists():
            return True, f"{final_csv} exists"
        if int(fold) == 1:
            single_fold_csv = exp_dir / "final_results.csv"
            if single_fold_csv.exists():
                return True, f"{single_fold_csv} exists"

    expected_epoch = int(run["epochs"])
    final_epoch_model = exp_dir / "models" / f"{expected_epoch}_model.pth"
    if final_epoch_model.exists():
        return True, f"{final_epoch_model} exists"

    return False, None


def is_run_completed(run: dict[str, Any]) -> bool:
    completed, _ = get_run_completion_status(run)
    return completed


def normalize_smoke_runs(
    runs: list[dict[str, Any]],
    smoke_limit: int | None,
) -> list[dict[str, Any]]:
    def smoke_output_dir(original_output_dir: object) -> str:
        output_dir = Path(str(original_output_dir if original_output_dir is not None else "output"))
        if output_dir == Path("output"):
            return "output_smoke"
        return str(Path("output_smoke") / output_dir.name)

    normalized_runs: list[dict[str, Any]] = []
    for run in runs:
        normalized_run = dict(run)
        normalized_run["epochs"] = 5
        normalized_run["target_fold"] = 1
        normalized_run["output_dir"] = smoke_output_dir(run.get("output_dir", "output"))
        normalized_run["max_train_batches"] = 1
        normalized_run["max_eval_batches"] = 1
        normalized_runs.append(normalized_run)
    if smoke_limit is None:
        return normalized_runs
    return normalized_runs[:smoke_limit]


def validate_and_warn_skip_completed_config(config: dict[str, Any], mode: str) -> None:
    deprecated_paths: list[str] = []
    if "skip_completed" in config:
        parse_bool_config(config, "skip_completed", default=False)
        deprecated_paths.append("skip_completed")

    if mode == "single":
        single_cfg = config.get("single", {})
        if isinstance(single_cfg, dict) and "skip_completed" in single_cfg:
            parse_bool_config(single_cfg, "skip_completed", default=False)
            deprecated_paths.append("single.skip_completed")
    elif mode == "multi":
        multi_cfg = config.get("multi", {})
        if isinstance(multi_cfg, dict) and "skip_completed" in multi_cfg:
            parse_bool_config(multi_cfg, "skip_completed", default=False)
            deprecated_paths.append("multi.skip_completed")

    if deprecated_paths:
        joined = ", ".join(deprecated_paths)
        print(
            "[WARN] Deprecated config key(s) "
            f"{joined} are ignored. Completed experiments are skipped automatically.",
            file=sys.stderr,
        )


def append_retried_experiments_log(
    repo_root: Path,
    config_path: Path,
    skipped_runs: list[dict[str, str]],
) -> Path | None:
    if not skipped_runs:
        return None

    log_path = repo_root / "output" / "retried_exp.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        config_label = config_path.relative_to(repo_root)
    except ValueError:
        config_label = config_path

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"## {timestamp} `{config_label}`\n")
        for skipped in skipped_runs:
            handle.write(
                "- "
                f"dataset=`{skipped['dataset']}`, "
                f"experiment=`{skipped['experiment_name']}`, "
                f"output_dir=`{skipped['output_dir']}`, "
                f"target_fold=`{skipped['target_fold']}`, "
                f"reason=`{skipped['reason']}`\n"
            )
        handle.write("\n")

    return log_path


def filter_completed_runs(
    runs: list[dict[str, Any]],
    repo_root: Path,
    config_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], Path | None]:
    kept_runs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, str]] = []

    for run in runs:
        completed, reason = get_run_completion_status(run)
        if not completed:
            kept_runs.append(run)
            continue

        experiment_name = get_run_experiment_name(run)
        model_tag = str(run.get("model", "dconnnet"))
        if model_tag != "dconnnet":
            experiment_name = f"{model_tag}/{experiment_name}"
        dataset = str(run["preset"]["dataset"])
        target_fold = run.get("target_fold")
        skipped = {
            "dataset": dataset,
            "experiment_name": experiment_name,
            "output_dir": str(get_run_output_root(run)),
            "target_fold": "-" if target_fold is None else str(target_fold),
            "reason": reason or "completed artifact detected",
        }
        skipped_runs.append(skipped)
        print(
            "[WARN] Completed experiment detected; skipping retried run: "
            f"dataset={dataset}, experiment={experiment_name}, "
            f"output_dir={skipped['output_dir']}, target_fold={skipped['target_fold']} "
            f"({skipped['reason']})",
            file=sys.stderr,
        )

    log_path = append_retried_experiments_log(repo_root, config_path, skipped_runs)
    return kept_runs, skipped_runs, log_path


def send_discord_alert(
    pybin: str,
    repo_root: Path,
    status: str,
    summary: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    alert_script = repo_root / "scripts" / "discord_alert.py"
    if not alert_script.exists():
        return
    subprocess.run(
        [
            pybin,
            str(alert_script),
            "--env-file",
            str(repo_root / ".env"),
            "--job",
            "train_launcher_config",
            "--status",
            status,
            "--summary",
            summary,
        ],
        check=False,
    )


def build_single_schedule(config: dict[str, Any], device: int) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    if dataset == "retouch" or str(dataset).startswith("retouch-"):
        raise ValueError("RETOUCH dataset is no longer supported by launcher")
    single_cfg = config.get("single") or {}
    if not isinstance(single_cfg, dict):
        raise ValueError("single block must be a mapping")
    reject_removed_decoder_fusion_keys(config, "top-level config")
    reject_removed_decoder_fusion_keys(single_cfg, "single config")

    model = str(single_cfg.get("model", config.get("model", "dconnnet")))
    conn_num = int(single_cfg.get("conn_num", 8))
    conn_layout = normalize_conn_layout(conn_num, single_cfg.get("conn_layout", config.get("conn_layout")))
    final_decision = normalize_final_decision(
        single_cfg.get("final_decision", config.get("final_decision", "max"))
    )
    conn_fusion = normalize_conn_fusion(single_cfg.get("conn_fusion", config.get("conn_fusion", "none")))
    fusion_lambda_fused = float(single_cfg.get("fusion_lambda_fused", config.get("fusion_lambda_fused", 0.3)))
    fusion_residual_scale = float(single_cfg.get("fusion_residual_scale", config.get("fusion_residual_scale", 0.2)))
    label_mode = normalize_label_mode(single_cfg.get("label_mode", "binary"))
    dist_aux_loss = str(single_cfg.get("dist_aux_loss", "smooth_l1"))
    use_seg_aux = parse_bool_config(
        single_cfg,
        "use_seg_aux",
        parse_bool_config(config, "use_seg_aux", False),
    )
    seg_aux_weight_raw = single_cfg.get("seg_aux_weight", config.get("seg_aux_weight"))
    seg_aux_weight = None if seg_aux_weight_raw is None else float(seg_aux_weight_raw)
    folds = int(single_cfg.get("folds", 1))
    training_ctrl_overrides = parse_training_control_overrides(single_cfg)
    test_only = parse_bool_config(single_cfg, "test_only", parse_bool_config(config, "test_only", False))
    simple_loss = parse_bool_config(
        single_cfg, "simple_loss",
        parse_bool_config(config, "simple_loss", False),
    )
    output_dir = str(single_cfg.get("output_dir", config.get("output_dir", "output")))
    experiment_name_suffix = str(single_cfg.get("experiment_name_suffix", config.get("experiment_name_suffix", "")) or "")
    pretrained_spec = single_cfg.get("pretrained", config.get("pretrained"))
    if pretrained_spec is not None and not isinstance(pretrained_spec, str):
        raise ValueError("pretrained must be a string path")
    fusion_branches = normalize_fusion_branches(single_cfg.get("fusion_branches", config.get("fusion_branches", "io")), "io")
    validate_final_decision_compatibility(
        final_decision,
        conn_layout,
        fusion_branches if conn_fusion != "none" else "none",
    )

    single_external_data_raw = config.get("external_data")
    single_external_data: str | None = None
    if single_external_data_raw is not None:
        single_external_data = str(single_external_data_raw)
        if single_external_data not in ALLOWED_EXTERNAL_DATASETS:
            raise ValueError(
                f"Unsupported external_data: {single_external_data} "
                f"(supported: {sorted(ALLOWED_EXTERNAL_DATASETS)})"
            )

    if use_seg_aux and seg_aux_weight is None:
        raise ValueError("single config uses use_seg_aux=true but does not set seg_aux_weight")

    batch_size = get_with_alias(
        single_cfg,
        ("batch_size", "batch-size"),
        get_with_alias(config, ("batch_size", "batch-size")),
    )

    if (dataset == "retouch" or dataset.startswith("retouch-")) and conn_num != 8:
        raise ValueError("retouch single mode currently supports conn_num=8 only")

    epochs = int(single_cfg.get("epochs", default_single_epochs(dataset, conn_num, label_mode)))

    if dataset in {"octa", "octa500"}:
        variant = str(single_cfg.get("octa_variant", "6M"))
        preset = resolve_preset(dataset, variant)
    elif dataset == "retouch" or dataset.startswith("retouch-"):
        # ... (handled in loop below)
        preset = None
    else:
        preset = resolve_preset(dataset)

    if preset is not None and batch_size is not None:
        preset = dict(preset)
        preset["batch_size"] = int(batch_size)

    def validate_single_layout_compatibility(active_preset: dict[str, Any]) -> None:
        if active_preset.get("num_class", 1) != 1 and conn_layout != "standard8":
            raise ValueError("single config supports non-standard conn_layout only for single-class runs")
        if conn_fusion != "none":
            if active_preset.get("num_class", 1) != 1:
                raise ValueError("conn_fusion supports only single-class runs")
            if conn_num != 8:
                raise ValueError("conn_fusion supports only conn_num=8")
            if conn_layout != "standard8":
                raise ValueError("conn_fusion supports only conn_layout=standard8")

    if dataset in {"octa", "octa500"}:
        target_folds = [None]
    elif dataset == "retouch" or dataset.startswith("retouch-"):
        if dataset.startswith("retouch-"):
            retouch_devices = [dataset.split("-", 1)[1]]
        else:
            retouch_devices = parse_retouch_devices(
                single_cfg.get("retouch_device"),
                ["Spectrailis"],
            )
        retouch_data_root = str(single_cfg.get("retouch_data_root", "data/retouch"))
        if folds != 3:
            raise ValueError("retouch single mode requires folds=3")
        target_folds_cfg = as_list(single_cfg.get("target_folds"), [1, 2, 3])
        target_folds = [int(v) for v in target_folds_cfg]
        for fold in target_folds:
            if fold < 1 or fold > 3:
                raise ValueError("retouch target_folds must be within [1, 3]")
        runs: list[dict[str, Any]] = []
        for retouch_device in retouch_devices:
            preset = resolve_preset(
                "retouch",
                retouch_device=retouch_device,
                retouch_data_root=retouch_data_root,
            )
            validate_single_layout_compatibility(preset)
            if batch_size is not None:
                preset = dict(preset)
                preset["batch_size"] = int(batch_size)
            for target_fold in target_folds:
                runs.append(
                    {
                        "preset": preset,
                        "conn_num": conn_num,
                        "conn_layout": conn_layout,
                        "final_decision": final_decision,
                        "label_mode": label_mode,
                        "dist_aux_loss": dist_aux_loss,
                        "conn_fusion": conn_fusion,
                        "fusion_branches": fusion_branches,
                        "fusion_lambda_fused": fusion_lambda_fused,
                        "fusion_residual_scale": fusion_residual_scale,
                        "epochs": epochs,
                        "folds": folds,
                        "target_fold": target_fold,
                        "device": device,
                        "training_ctrl_overrides": training_ctrl_overrides,
                        "output_dir": output_dir,
                        "experiment_name_suffix": experiment_name_suffix,
                        "test_only": True if single_external_data else test_only,
                        "simple_loss": simple_loss,
                        "pretrained": resolve_pretrained_path(
                            pretrained_spec=pretrained_spec,
                            output_dir=output_dir,
                            dataset=str(preset["dataset"]),
                            conn_num=conn_num,
                            label_mode=label_mode,
                            dist_aux_loss=dist_aux_loss,
                            conn_layout=conn_layout,
                            conn_fusion=conn_fusion,
                            fusion_residual_scale=fusion_residual_scale,
                            use_seg_aux=use_seg_aux,
                            seg_aux_weight=seg_aux_weight,
                            experiment_name_suffix=experiment_name_suffix,
                            fusion_branches=fusion_branches,
                            final_decision=final_decision,
                        ) if (test_only or single_external_data) else None,
                        "use_seg_aux": use_seg_aux,
                        "seg_aux_weight": seg_aux_weight,
                        "model": model,
                        "external_data": single_external_data,
                    }
                )
        return runs
    else:
        target_folds = [None]

    validate_single_layout_compatibility(preset)

    single_run_preset = resolve_external_preset(single_external_data) if single_external_data else preset
    single_run_source_dataset = str(preset["dataset"])
    return [
        {
            "preset": single_run_preset,
            "conn_num": conn_num,
            "conn_layout": conn_layout,
            "final_decision": final_decision,
            "label_mode": label_mode,
            "dist_aux_loss": dist_aux_loss,
            "conn_fusion": conn_fusion,
            "fusion_branches": fusion_branches,
            "fusion_lambda_fused": fusion_lambda_fused,
            "fusion_residual_scale": fusion_residual_scale,
            "epochs": epochs,
            "folds": folds,
            "target_fold": target_folds[0],
            "device": device,
            "training_ctrl_overrides": training_ctrl_overrides,
            "output_dir": output_dir,
            "experiment_name_suffix": experiment_name_suffix,
            "test_only": True if single_external_data else test_only,
            "simple_loss": simple_loss,
            "pretrained": resolve_pretrained_path(
                pretrained_spec=pretrained_spec,
                output_dir=output_dir,
                dataset=single_run_source_dataset,
                conn_num=conn_num,
                label_mode=label_mode,
                dist_aux_loss=dist_aux_loss,
                conn_layout=conn_layout,
                conn_fusion=conn_fusion,
                fusion_residual_scale=fusion_residual_scale,
                use_seg_aux=use_seg_aux,
                seg_aux_weight=seg_aux_weight,
                experiment_name_suffix=experiment_name_suffix,
                fusion_branches=fusion_branches,
                final_decision=final_decision,
            ) if (test_only or single_external_data) else None,
            "use_seg_aux": use_seg_aux,
            "seg_aux_weight": seg_aux_weight,
            "model": model,
            "external_data": single_external_data,
        }
    ]


def build_multi_schedule(config: dict[str, Any], device: int, datasets: list[str]) -> list[dict[str, Any]]:
    if any(dataset == "retouch" or str(dataset).startswith("retouch-") for dataset in datasets):
        raise ValueError("RETOUCH dataset is no longer supported by launcher")
    multi_cfg = config.get("multi") or {}
    if not isinstance(multi_cfg, dict):
        raise ValueError("multi block must be a mapping")
    reject_removed_decoder_fusion_keys(config, "top-level config")
    reject_removed_decoder_fusion_keys(multi_cfg, "multi config")

    epochs = int(multi_cfg.get("epochs", int(os.getenv("MULTI_TRAIN_EPOCHS", "500"))))
    folds = int(multi_cfg.get("folds", int(os.getenv("MULTI_TRAIN_FOLDS", "1"))))
    training_ctrl_overrides = parse_training_control_overrides(multi_cfg)
    test_only = parse_bool_config(multi_cfg, "test_only", parse_bool_config(config, "test_only", False))
    output_dir = str(multi_cfg.get("output_dir", config.get("output_dir", "output")))
    multi_experiment_name_suffix = str(
        multi_cfg.get("experiment_name_suffix", config.get("experiment_name_suffix", "")) or ""
    )
    pretrained_spec = multi_cfg.get("pretrained", config.get("pretrained"))
    if pretrained_spec is not None and not isinstance(pretrained_spec, str):
        raise ValueError("pretrained must be a string path")

    external_data_raw = config.get("external_data")
    external_data: str | None = None
    if external_data_raw is not None:
        external_data = str(external_data_raw)
        if external_data not in ALLOWED_EXTERNAL_DATASETS:
            raise ValueError(
                f"Unsupported external_data: {external_data} "
                f"(supported: {sorted(ALLOWED_EXTERNAL_DATASETS)})"
            )
    multi_fusion_branches = normalize_fusion_branches(multi_cfg.get("fusion_branches", config.get("fusion_branches", "io")), "io")
    final_decisions_raw = get_with_alias(
        multi_cfg,
        ("final_decisions", "final_decision"),
        config.get("final_decision", "max"),
    )
    final_decisions = [
        normalize_final_decision(value)
        for value in as_list(final_decisions_raw, ["max"])
    ]
    if not final_decisions:
        raise ValueError("multi.final_decisions must be non-empty")

    batch_size = get_with_alias(
        multi_cfg,
        ("batch_size", "batch-size"),
        get_with_alias(config, ("batch_size", "batch-size")),
    )

    conn_values = [int(v) for v in as_list(get_with_alias(multi_cfg, ("conn_nums", "conn_num")), [8, 24])]
    conn_layouts_raw = as_list(get_with_alias(multi_cfg, ("conn_layouts", "conn_layout")), [None])
    conn_fusions = [
        normalize_conn_fusion(v)
        for v in as_list(get_with_alias(multi_cfg, ("conn_fusions", "conn_fusion")), ["none"])
    ]

    def _parse_multi_float_sweep(
        keys: tuple[str, ...],
        default_value: float,
        label: str,
    ) -> list[float]:
        raw = get_with_alias(multi_cfg, keys)
        if raw is None:
            return [float(default_value)]
        values = [float(v) for v in as_list(raw, [])]
        if not values:
            raise ValueError(f"{label} must be a non-empty list when provided")
        return values

    fusion_lambda_fused = float(get_with_alias(multi_cfg, ("fusion_lambda_fused",), config.get("fusion_lambda_fused", 0.3)))
    conn_aux_c3_weights = _parse_multi_float_sweep(
        ("conn_aux_c3_weights", "conn_aux_c3_weight"),
        0.3,
        "multi.conn_aux_c3_weights",
    )
    conn_aux_c4_weights = _parse_multi_float_sweep(
        ("conn_aux_c4_weights", "conn_aux_c4_weight"),
        0.2,
        "multi.conn_aux_c4_weights",
    )
    conn_aux_c5_weights = _parse_multi_float_sweep(
        ("conn_aux_c5_weights", "conn_aux_c5_weight"),
        0.2,
        "multi.conn_aux_c5_weights",
    )
    conn_aux_c6_weights = _parse_multi_float_sweep(
        ("conn_aux_c6_weights", "conn_aux_c6_weight"),
        0.3,
        "multi.conn_aux_c6_weights",
    )
    conn_aux_c7_weights = _parse_multi_float_sweep(
        ("conn_aux_c7_weights", "conn_aux_c7_weight"),
        0.2,
        "multi.conn_aux_c7_weights",
    )
    default_residual_scale = float(get_with_alias(multi_cfg, ("fusion_residual_scale",), config.get("fusion_residual_scale", 0.2)))
    fusion_residual_scales = [
        float(v)
        for v in as_list(get_with_alias(multi_cfg, ("fusion_residual_scales", "fusion_residual_scale")), [default_residual_scale])
    ]
    fusion_matrix_cfg = multi_cfg.get("fusion_matrix")
    fusion_grid: list[dict[str, Any]] = []
    if fusion_matrix_cfg is not None:
        if not isinstance(fusion_matrix_cfg, list) or not fusion_matrix_cfg:
            raise ValueError("multi.fusion_matrix must be a non-empty list")
        for idx, entry in enumerate(fusion_matrix_cfg):
            if not isinstance(entry, dict):
                raise ValueError(f"multi.fusion_matrix[{idx}] must be a mapping")
            reject_removed_decoder_fusion_keys(entry, f"multi.fusion_matrix[{idx}]")
            if "conn_fusion" not in entry:
                raise ValueError(f"multi.fusion_matrix[{idx}] requires `conn_fusion`")
            conn_fusion = normalize_conn_fusion(entry.get("conn_fusion"))
            entry_default_residual_scale = float(
                get_with_alias(entry, ("fusion_residual_scale",), default_residual_scale)
            )
            entry_residual_scales = [
                float(v)
                for v in as_list(
                    get_with_alias(entry, ("fusion_residual_scales", "fusion_residual_scale")),
                    [entry_default_residual_scale],
                )
            ]
            if conn_fusion == "none":
                entry_residual_scales = [entry_default_residual_scale]
            elif conn_fusion == "scaled_sum":
                pass
            else:
                if "fusion_residual_scales" in entry and len(entry_residual_scales) > 1:
                    raise ValueError(
                        f"multi.fusion_matrix[{idx}] uses fusion_residual_scales, "
                        "but only scaled_sum supports residual scale sweeps"
                    )
                entry_residual_scales = [entry_default_residual_scale]

            entry_c5_raw = get_with_alias(entry, ("conn_aux_c5_weights", "conn_aux_c5_weight"))
            entry_c5_weights = (
                [float(v) for v in as_list(entry_c5_raw, [])]
                if entry_c5_raw is not None
                else None
            )
            if entry_c5_weights is not None and not entry_c5_weights:
                raise ValueError(f"multi.fusion_matrix[{idx}].conn_aux_c5_weights must be non-empty")

            fusion_grid.append(
                {
                    "conn_fusion": conn_fusion,
                    "fusion_residual_scales": entry_residual_scales,
                    "conn_aux_c5_weights": entry_c5_weights,
                }
            )
    else:
        for conn_fusion in conn_fusions:
            residual_scales = (
                fusion_residual_scales
                if conn_fusion == "scaled_sum"
                else [default_residual_scale]
            )
            fusion_grid.append(
                {
                    "conn_fusion": conn_fusion,
                    "fusion_residual_scales": residual_scales,
                    "conn_aux_c5_weights": None,
                }
            )

    label_modes = [
        normalize_label_mode(v)
        for v in as_list(get_with_alias(multi_cfg, ("label_modes", "label_mode")), ["binary", "dist", "dist_inverted"])
    ]
    dist_aux_values = [str(v) for v in as_list(get_with_alias(multi_cfg, ("dist_aux_losses", "dist_aux_loss")), ["gjml_sf_l1", "smooth_l1"])]
    binary_dist_aux = str(get_with_alias(multi_cfg, ("binary_dist_aux_loss",), "smooth_l1"))

    multi_use_seg_aux = parse_bool_config(multi_cfg, "use_seg_aux", default=False)
    seg_aux_weights_raw = get_with_alias(multi_cfg, ("seg_aux_weights", "seg_aux_weight"))
    if seg_aux_weights_raw is None:
        if multi_use_seg_aux:
            raise ValueError("multi.use_seg_aux=true requires multi.seg_aux_weights to be set")
        seg_aux_weights: list[float | None] = [None]
    else:
        if not multi_use_seg_aux:
            raise ValueError(
                "multi.seg_aux_weights is set but multi.use_seg_aux is false; "
                "set use_seg_aux=true to enable the seg_aux sweep"
            )
        seg_aux_weights = [float(v) for v in as_list(seg_aux_weights_raw, [])]
        if not seg_aux_weights:
            raise ValueError("multi.seg_aux_weights must be a non-empty list when use_seg_aux=true")

    multi_default_simple_loss = parse_bool_config(multi_cfg, "simple_loss", default=False)
    experiments = multi_cfg.get("experiments")
    experiments_only = parse_bool_config(multi_cfg, "experiments_only", default=False)
    if experiments_only and not experiments:
        raise ValueError("multi.experiments_only requires multi.experiments to be set")

    runs: list[dict[str, Any]] = []
    grid_datasets = [] if experiments_only else datasets
    for dataset in grid_datasets:
        dataset_conn_values = list(conn_values)
        if dataset in {"octa", "octa500"}:
            default_variants = os.getenv("MULTI_OCTA_VARIANTS", "3M 6M").replace(",", " ").split()
            octa_variants = [str(v) for v in as_list(multi_cfg.get("octa_variants"), default_variants)]
            retouch_devices = [None]
            retouch_folds = [None]
        elif dataset == "retouch" or dataset.startswith("retouch-"):
            if dataset.startswith("retouch-"):
                retouch_devices = [dataset.split("-", 1)[1]]
            else:
                retouch_devices = parse_retouch_devices(
                    multi_cfg.get("retouch_devices"),
                    ["Cirrus", "Spectrailis", "Topcon"],
                )
            retouch_folds_cfg = as_list(multi_cfg.get("retouch_target_folds"), [1, 2, 3])
            retouch_folds = [int(v) for v in retouch_folds_cfg]
            if folds != 3:
                raise ValueError("retouch multi mode requires folds=3")
            if conn_values != [8]:
                print(
                    "[INFO] retouch multi-class path currently supports conn_num=8; forcing conn sweep to [8]",
                    file=sys.stderr,
                )
            dataset_conn_values = [8]
            for fold in retouch_folds:
                if fold < 1 or fold > 3:
                    raise ValueError("retouch_target_folds must be within [1, 3]")
            octa_variants = [None]
        else:
            octa_variants = [None]
            retouch_devices = [None]
            retouch_folds = [None]

        for variant in octa_variants:
            for retouch_device in retouch_devices:
                if dataset == "retouch" or dataset.startswith("retouch-"):
                    retouch_data_root = str(multi_cfg.get("retouch_data_root", "data/retouch"))
                    preset = resolve_preset(
                        "retouch",
                        retouch_device=retouch_device,
                        retouch_data_root=retouch_data_root,
                    )
                else:
                    preset = resolve_preset(dataset, variant)

                if batch_size is not None:
                    preset = dict(preset)
                    preset["batch_size"] = int(batch_size)
                for label_mode in label_modes:
                    aux_losses = [binary_dist_aux] if label_mode == "binary" else dist_aux_values
                    for conn_num in dataset_conn_values:
                        conn_layouts = [normalize_conn_layout(conn_num, value) for value in conn_layouts_raw]
                        if preset.get("num_class", 1) != 1:
                            invalid = [layout for layout in conn_layouts if layout != "standard8"]
                            if invalid:
                                raise ValueError("multi-class runs support only conn_layout=standard8")
                            invalid_fusions = [
                                str(grid_entry["conn_fusion"])
                                for grid_entry in fusion_grid
                                if str(grid_entry["conn_fusion"]) != "none"
                            ]
                            if invalid_fusions:
                                raise ValueError("conn_fusion supports only single-class runs")
                        for dist_aux_loss in aux_losses:
                            for conn_layout in conn_layouts:
                                for fusion_cfg in fusion_grid:
                                    conn_fusion = str(fusion_cfg["conn_fusion"])
                                    if conn_fusion != "none":
                                        if conn_num != 8:
                                            raise ValueError("conn_fusion supports only conn_num=8")
                                        if conn_layout != "standard8":
                                            raise ValueError("conn_fusion supports only conn_layout=standard8")
                                    residual_scales = [float(v) for v in fusion_cfg["fusion_residual_scales"]]
                                    effective_c5_weights = fusion_cfg["conn_aux_c5_weights"] or conn_aux_c5_weights
                                    for fusion_residual_scale in residual_scales:
                                        for target_fold in retouch_folds:
                                            for (
                                                final_decision,
                                                seg_aux_weight,
                                                c3w_v,
                                                c4w_v,
                                                c5w_v,
                                                c6w_v,
                                                c7w_v,
                                            ) in itertools.product(
                                                final_decisions,
                                                seg_aux_weights,
                                                conn_aux_c3_weights,
                                                conn_aux_c4_weights,
                                                effective_c5_weights,
                                                conn_aux_c6_weights,
                                                conn_aux_c7_weights,
                                            ):
                                                validate_final_decision_compatibility(
                                                    final_decision,
                                                    conn_layout,
                                                    multi_fusion_branches if conn_fusion != "none" else "none",
                                                )
                                                sweep_tag_parts: list[str] = []
                                                if len(conn_aux_c3_weights) > 1:
                                                    sweep_tag_parts.append(f"c3-{c3w_v:g}")
                                                if len(conn_aux_c4_weights) > 1:
                                                    sweep_tag_parts.append(f"c4-{c4w_v:g}")
                                                if len(effective_c5_weights) > 1:
                                                    c5_str = f"{c5w_v:g}"
                                                    if "." not in c5_str and "e" not in c5_str:
                                                        c5_str += ".0"
                                                    sweep_tag_parts.append(f"c5-{c5_str}")
                                                if len(conn_aux_c6_weights) > 1:
                                                    sweep_tag_parts.append(f"c6-{c6w_v:g}")
                                                if len(conn_aux_c7_weights) > 1:
                                                    sweep_tag_parts.append(f"c7-{c7w_v:g}")
                                                if multi_experiment_name_suffix:
                                                    sweep_tag_parts.append(multi_experiment_name_suffix)
                                                run_experiment_name_suffix = "_".join(sweep_tag_parts)
                                                if external_data:
                                                    run_preset = resolve_external_preset(external_data)
                                                    run_test_only = True
                                                    run_pretrained = resolve_pretrained_path(
                                                        pretrained_spec=pretrained_spec,
                                                        output_dir=output_dir,
                                                        dataset=dataset,
                                                        conn_num=conn_num,
                                                        label_mode=label_mode,
                                                        dist_aux_loss=dist_aux_loss,
                                                        conn_layout=conn_layout,
                                                        conn_fusion=conn_fusion,
                                                        fusion_residual_scale=fusion_residual_scale,
                                                        use_seg_aux=multi_use_seg_aux,
                                                        seg_aux_weight=seg_aux_weight,
                                                        experiment_name_suffix=run_experiment_name_suffix or None,
                                                        fusion_branches=multi_fusion_branches,
                                                        final_decision=final_decision,
                                                    )
                                                else:
                                                    run_preset = preset
                                                    run_test_only = test_only
                                                    run_pretrained = resolve_pretrained_path(
                                                        pretrained_spec=pretrained_spec,
                                                        output_dir=output_dir,
                                                        dataset=str(preset["dataset"]),
                                                        conn_num=conn_num,
                                                        label_mode=label_mode,
                                                        dist_aux_loss=dist_aux_loss,
                                                        conn_layout=conn_layout,
                                                        conn_fusion=conn_fusion,
                                                        fusion_residual_scale=fusion_residual_scale,
                                                        use_seg_aux=multi_use_seg_aux,
                                                        seg_aux_weight=seg_aux_weight,
                                                        experiment_name_suffix=run_experiment_name_suffix or None,
                                                        fusion_branches=multi_fusion_branches,
                                                        final_decision=final_decision,
                                                    ) if test_only else None
                                                runs.append(
                                                    {
                                                        "preset": run_preset,
                                                        "conn_num": conn_num,
                                                        "conn_layout": conn_layout,
                                                        "final_decision": final_decision,
                                                        "label_mode": label_mode,
                                                        "dist_aux_loss": dist_aux_loss,
                                                        "conn_fusion": conn_fusion,
                                                        "fusion_branches": multi_fusion_branches,
                                                        "fusion_lambda_fused": fusion_lambda_fused,
                                                        "fusion_residual_scale": fusion_residual_scale,
                                                        "conn_aux_c3_weight": c3w_v,
                                                        "conn_aux_c4_weight": c4w_v,
                                                        "conn_aux_c5_weight": c5w_v,
                                                        "conn_aux_c6_weight": c6w_v,
                                                        "conn_aux_c7_weight": c7w_v,
                                                        "use_seg_aux": multi_use_seg_aux,
                                                        "seg_aux_weight": seg_aux_weight,
                                                        "epochs": epochs,
                                                        "folds": folds,
                                                        "target_fold": target_fold,
                                                        "device": device,
                                                        "training_ctrl_overrides": training_ctrl_overrides,
                                                        "output_dir": output_dir,
                                                        "experiment_name_suffix": run_experiment_name_suffix,
                                                        "test_only": run_test_only,
                                                        "pretrained": run_pretrained,
                                                        "external_data": external_data,
                                                    }
                                                )
    # Handle explicit experiments list if present
    if experiments:
        if not isinstance(experiments, list):
            raise ValueError("multi.experiments must be a list")
        for exp_idx, exp_cfg in enumerate(experiments):
            if not isinstance(exp_cfg, dict):
                raise ValueError(f"multi.experiments[{exp_idx}] must be a mapping")
            reject_removed_decoder_fusion_keys(exp_cfg, f"multi.experiments[{exp_idx}]")

            exp_datasets = _normalize_experiment_datasets(exp_cfg, default_datasets=datasets)
            for dataset in exp_datasets:
                if dataset in {"octa", "octa500"}:
                    octa_variant = str(exp_cfg.get("octa_variant", "6M"))
                    preset = resolve_preset(dataset, octa_variant=octa_variant)
                elif dataset == "retouch" or dataset.startswith("retouch-"):
                    if dataset.startswith("retouch-"):
                        retouch_device = dataset.split("-", 1)[1]
                    else:
                        retouch_device = str(exp_cfg.get("retouch_device", "Spectrailis"))
                    retouch_data_root = str(exp_cfg.get("retouch_data_root", multi_cfg.get("retouch_data_root", "data/retouch")))
                    preset = resolve_preset(
                        "retouch",
                        retouch_device=retouch_device,
                        retouch_data_root=retouch_data_root,
                    )
                else:
                    preset = resolve_preset(dataset)

                exp_batch_size = exp_cfg.get("batch_size")
                if exp_batch_size is not None:
                    preset = dict(preset)
                    preset["batch_size"] = int(exp_batch_size)
                elif batch_size is not None:
                    preset = dict(preset)
                    preset["batch_size"] = int(batch_size)

                exp_conn_num = int(exp_cfg.get("conn_num", conn_values[0] if conn_values else 8))
                exp_label_mode = normalize_label_mode(
                    exp_cfg.get("label_mode", label_modes[0] if label_modes else "binary")
                )
                if "dist_aux_loss" in exp_cfg and exp_cfg["dist_aux_loss"] is not None:
                    exp_dist_aux_loss = str(exp_cfg["dist_aux_loss"])
                else:
                    exp_dist_aux_loss = binary_dist_aux if exp_label_mode == "binary" else (dist_aux_values[0] if dist_aux_values else "smooth_l1")
                exp_conn_layout = normalize_conn_layout(exp_conn_num, exp_cfg.get("conn_layout", "standard8"))
                exp_final_decision = normalize_final_decision(
                    exp_cfg.get("final_decision", final_decisions[0])
                )
                exp_conn_fusion = normalize_conn_fusion(exp_cfg.get("conn_fusion", "none"))

                if exp_conn_fusion != "none":
                    if exp_conn_num != 8:
                        raise ValueError("multi.experiments with conn_fusion requires conn_num=8")
                    if exp_conn_layout != "standard8":
                        raise ValueError("multi.experiments with conn_fusion requires conn_layout=standard8")

                exp_fusion_branches = normalize_fusion_branches(exp_cfg.get("fusion_branches", "io"), "io")
                validate_final_decision_compatibility(
                    exp_final_decision,
                    exp_conn_layout,
                    exp_fusion_branches if exp_conn_fusion != "none" else "none",
                )
                exp_use_seg_aux = bool(exp_cfg.get("use_seg_aux", False))
                exp_seg_aux_weight = (
                    None
                    if exp_cfg.get("seg_aux_weight") is None
                    else float(exp_cfg.get("seg_aux_weight"))
                )
                if exp_use_seg_aux and exp_seg_aux_weight is None:
                    raise ValueError(
                        f"multi.experiments[{exp_idx}] uses use_seg_aux=true "
                        "but does not set seg_aux_weight"
                    )
                exp_simple_loss = bool(exp_cfg.get("simple_loss", multi_default_simple_loss))
                exp_model = str(exp_cfg.get("model", "dconnnet"))
                exp_num_channels = (
                    None if exp_cfg.get("num_channels") is None
                    else int(exp_cfg["num_channels"])
                )
                exp_diff_timesteps = int(exp_cfg.get("diff_timesteps", 100))
                exp_diff_objective = str(exp_cfg.get("diff_objective", "pred_noise"))
                exp_beta_schedule = str(exp_cfg.get("beta_schedule", "linear"))
                exp_image_size = int(exp_cfg.get("image_size", 128))
                exp_device = int(exp_cfg.get("device", device))
                exp_output_dir = str(exp_cfg.get("output_dir", output_dir))
                exp_experiment_name_suffix = str(exp_cfg.get("experiment_name_suffix", "") or "")
                runs.append(
                    {
                        "preset": resolve_external_preset(external_data) if external_data else preset,
                        "conn_num": exp_conn_num,
                        "conn_layout": exp_conn_layout,
                        "final_decision": exp_final_decision,
                        "label_mode": exp_label_mode,
                        "dist_aux_loss": exp_dist_aux_loss,
                        "conn_fusion": exp_conn_fusion,
                        "fusion_branches": exp_fusion_branches,
                        "fusion_lambda_middle": float(exp_cfg.get("fusion_lambda_middle", 0.05)),
                        "fusion_lambda_fused": float(exp_cfg.get("fusion_lambda_fused", fusion_lambda_fused)),
                        "fusion_residual_scale": float(exp_cfg.get("fusion_residual_scale", default_residual_scale)),
                        "use_seg_aux": exp_use_seg_aux,
                        "seg_aux_weight": exp_seg_aux_weight,
                        "fusion_gate_reg_weight": float(exp_cfg.get("fusion_gate_reg_weight", 0.01)),
                        "conn_aux_c3_weight": float(exp_cfg.get("conn_aux_c3_weight", 0.3)),
                        "conn_aux_c4_weight": float(exp_cfg.get("conn_aux_c4_weight", 0.2)),
                        "conn_aux_c5_weight": float(exp_cfg.get("conn_aux_c5_weight", 0.2)),
                        "conn_aux_c6_weight": float(exp_cfg.get("conn_aux_c6_weight", 0.3)),
                        "conn_aux_c7_weight": float(exp_cfg.get("conn_aux_c7_weight", 0.2)),
                        "legacy_aux_weight": float(
                            exp_cfg.get(
                                "legacy_aux_weight",
                                multi_cfg.get("legacy_aux_weight", config.get("legacy_aux_weight", 0.0)),
                            )
                        ),
                        "debug_loss_breakdown": bool(
                            exp_cfg.get(
                                "debug_loss_breakdown",
                                multi_cfg.get(
                                    "debug_loss_breakdown",
                                    config.get("debug_loss_breakdown", False),
                                ),
                            )
                        ),
                        "epochs": int(exp_cfg.get("epochs", epochs)),
                        "save_per_epochs": (
                            int(exp_cfg["save_interval"])
                            if exp_cfg.get("save_interval") is not None
                            else None
                        ),
                        "folds": int(exp_cfg.get("folds", folds)),
                        "target_fold": None,
                        "device": exp_device,
                        "training_ctrl_overrides": training_ctrl_overrides,
                        "output_dir": exp_output_dir,
                        "experiment_name_suffix": exp_experiment_name_suffix,
                        "test_only": True if external_data else test_only,
                        "simple_loss": exp_simple_loss,
                        "model": exp_model,
                        "num_channels": exp_num_channels,
                        "diff_timesteps": exp_diff_timesteps,
                        "diff_objective": exp_diff_objective,
                        "beta_schedule": exp_beta_schedule,
                        "image_size": exp_image_size,
                        "pretrained": resolve_pretrained_path(
                            pretrained_spec=pretrained_spec,
                            output_dir=exp_output_dir,
                            dataset=str(preset["dataset"]),
                            conn_num=exp_conn_num,
                            label_mode=exp_label_mode,
                            dist_aux_loss=exp_dist_aux_loss,
                            conn_layout=exp_conn_layout,
                            conn_fusion=exp_conn_fusion,
                            fusion_residual_scale=float(exp_cfg.get("fusion_residual_scale", default_residual_scale)),
                            use_seg_aux=exp_use_seg_aux,
                            seg_aux_weight=exp_seg_aux_weight,
                            experiment_name_suffix=exp_experiment_name_suffix,
                            fusion_branches=exp_fusion_branches,
                            final_decision=exp_final_decision,
                        ) if (test_only or external_data) else None,
                        "external_data": external_data,
                    }
                )

    return runs


def validate_config_shape(config: dict[str, Any]) -> list[str]:
    removed_keys = [key for key in ("direction_grouping", "direction_fusion") if key in config]
    if removed_keys:
        raise ValueError(
            "Unsupported config key(s): "
            + ", ".join(sorted(removed_keys))
            + ". Direction-grouping options were removed from this fork."
        )

    dataset_value = config.get("dataset")
    mode = config.get("mode")

    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported mode: {mode} (supported: {sorted(ALLOWED_MODES)})")
    datasets = normalize_datasets_for_mode(dataset_value, mode)

    unsupported_datasets = sorted({dataset for dataset in datasets if dataset not in ALLOWED_DATASETS})
    if unsupported_datasets:
        raise ValueError(
            f"Unsupported dataset(s): {unsupported_datasets} "
            f"(supported: {sorted(ALLOWED_DATASETS)})"
        )

    return datasets


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    status = "FAILED"
    summary = f"launcher failed: {config_path.name}"

    try:
        config = load_config(config_path)
        datasets = validate_config_shape(config)

        if args.device is not None:
            config["device"] = int(args.device)
        if args.test_only:
            config["test_only"] = True
        if args.pretrained is not None:
            config["pretrained"] = None if args.pretrained == "__AUTO__" else str(args.pretrained)
        if args.batch_size is not None:
            config["batch_size"] = int(args.batch_size)

        mode = config["mode"]
        device = int(config.get("device", 0))
        dist_sf_l1_gamma = float(config.get("dist_sf_l1_gamma", 1.0))

        if args.smoke:
            apply_smoke_overrides(config, mode)
            if args.batch_size is None:
                config["batch_size"] = 1
            if args.smoke_limit is not None and args.smoke_limit < 1:
                raise ValueError("--smoke_limit must be >= 1")

        if "isic2018" in datasets:
            ensure_isic_dataset_ready(repo_root)

        runs: list[dict[str, Any]] = []
        if mode == "single":
            config["dataset"] = datasets[0]
            runs = build_single_schedule(config, device)
        else:
            runs = build_multi_schedule(config, device, datasets)

        if args.smoke:
            runs = normalize_smoke_runs(runs, args.smoke_limit)

        validate_and_warn_skip_completed_config(config, mode)
        runs, skipped_runs, retry_log_path = filter_completed_runs(runs, repo_root, config_path)
        if skipped_runs:
            retry_log_msg = "" if retry_log_path is None else f" Logged to {retry_log_path}."
            print(
                f"[WARN] Skipped {len(skipped_runs)} completed run(s).{retry_log_msg}",
                file=sys.stderr,
            )

        pybin = sys.executable
        run_metric_lines: list[str] = []
        for run in runs:
            if run.get("test_only") and run.get("pretrained") and not args.dry_run:
                if not Path(str(run["pretrained"])).exists():
                    raise ValueError(f"Missing pretrained checkpoint for test_only: {run['pretrained']}")
            cmd = build_train_cmd(
                pybin=pybin,
                repo_root=repo_root,
                preset=run["preset"],
                model=str(run.get("model", "dconnnet")),
                conn_num=run["conn_num"],
                conn_layout=run["conn_layout"],
                final_decision=normalize_final_decision(run.get("final_decision", "max")),
                label_mode=run["label_mode"],
                dist_aux_loss=run["dist_aux_loss"],
                dist_sf_l1_gamma=dist_sf_l1_gamma,
                device=run["device"],
                epochs=run["epochs"],
                folds=run["folds"],
                target_fold=run.get("target_fold"),
                training_ctrl_overrides=run.get("training_ctrl_overrides"),
                output_dir=run.get("output_dir"),
                pretrained=run.get("pretrained"),
                test_only=bool(run.get("test_only", False)),
                conn_fusion=str(run.get("conn_fusion", "none")),
                fusion_branches=normalize_fusion_branches(run.get("fusion_branches", "io"), "io"),
                fusion_lambda_middle=float(run.get("fusion_lambda_middle", 0.05)),
                fusion_lambda_fused=float(run.get("fusion_lambda_fused", 0.3)),
                fusion_residual_scale=float(run.get("fusion_residual_scale", 0.2)),
                use_seg_aux=bool(run.get("use_seg_aux", False)),
                seg_aux_weight=(
                    None
                    if run.get("seg_aux_weight") is None
                    else float(run.get("seg_aux_weight"))
                ),
                fusion_gate_reg_weight=float(run.get("fusion_gate_reg_weight", 0.01)),
                conn_aux_c3_weight=float(run.get("conn_aux_c3_weight", 0.3)),
                conn_aux_c4_weight=float(run.get("conn_aux_c4_weight", 0.2)),
                conn_aux_c5_weight=float(run.get("conn_aux_c5_weight", 0.2)),
                conn_aux_c6_weight=float(run.get("conn_aux_c6_weight", 0.3)),
                conn_aux_c7_weight=float(run.get("conn_aux_c7_weight", 0.2)),
                legacy_aux_weight=float(run.get("legacy_aux_weight", 0.0)),
                debug_loss_breakdown=bool(run.get("debug_loss_breakdown", False)),
                experiment_name_suffix=str(run.get("experiment_name_suffix", "") or ""),
                max_train_batches=(
                    None
                    if run.get("max_train_batches") is None
                    else int(run.get("max_train_batches"))
                ),
                max_eval_batches=(
                    None
                    if run.get("max_eval_batches") is None
                    else int(run.get("max_eval_batches"))
                ),
                simple_loss=bool(run.get("simple_loss", False)),
                num_channels=(
                    None if run.get("num_channels") is None
                    else int(run["num_channels"])
                ),
                diff_timesteps=int(run.get("diff_timesteps", 100)),
                diff_objective=str(run.get("diff_objective", "pred_noise")),
                beta_schedule=str(run.get("beta_schedule", "linear")),
                image_size=int(run.get("image_size", 128)),
                save_per_epochs=(
                    None if run.get("save_per_epochs") is None
                    else int(run["save_per_epochs"])
                ),
                external_data=run.get("external_data"),
            )
            run_cmd(cmd, args.dry_run)
            metrics = read_run_final_metrics(run)
            run_metric_lines.append(format_run_metrics_line(run, metrics))

        run_count = len(runs)
        dataset_summary = format_dataset_summary(datasets)
        print(
            f"[INFO] Completed {mode} schedule for dataset(s)={dataset_summary} "
            f"with {run_count} run(s).",
            file=sys.stderr,
        )
        status = "DONE"
        base_summary = f"launcher done: dataset={dataset_summary}, mode={mode}, runs={run_count}"
        if run_metric_lines:
            summary = base_summary + "\n\nResults:\n" + "\n\n".join(run_metric_lines)
        else:
            summary = base_summary
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        summary = f"launcher failed: {config_path.name}\n\nError: {exc}"
        return 2
    finally:
        send_discord_alert(
            pybin=sys.executable,
            repo_root=repo_root,
            status=status,
            summary=summary,
            dry_run=args.dry_run or args.smoke,
        )


if __name__ == "__main__":
    raise SystemExit(main())
