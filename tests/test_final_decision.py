from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from connect_loss import Bilateral_voting, connect_loss
from scripts import train_launcher_from_config as launcher
from src.losses.factory import build_loss_functions


def _patch_cuda_to_noop(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False)


def _identity_translation(size):
    return torch.eye(size, dtype=torch.float32).view(1, 1, size, size)


def _loss_args(final_decision="max", conn_layout="standard8"):
    return SimpleNamespace(
        sigma=2.0,
        tau=3.0,
        dist_aux_loss="smooth_l1",
        dist_sf_l1_gamma=1.0,
        num_class=1,
        conn_num=8,
        conn_layout=conn_layout,
        final_decision=final_decision,
    )


def test_bilateral_voting_reductions_return_4d_masks(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)

    c_map = torch.stack(
        [torch.full((2, 2), float(channel)) for channel in range(1, 9)],
        dim=0,
    ).view(1, 1, 8, 2, 2)
    translation = _identity_translation(2)

    pred_max, vote_max = Bilateral_voting(c_map, translation, translation, "max")
    pred_mean, vote_mean = Bilateral_voting(c_map, translation, translation, "mean")
    pred_weighted, vote_weighted = Bilateral_voting(
        c_map, translation, translation, "weighted_mean"
    )

    assert pred_max.shape == (1, 1, 2, 2)
    assert pred_mean.shape == (1, 1, 2, 2)
    assert pred_weighted.shape == (1, 1, 2, 2)
    assert vote_max.shape == vote_mean.shape == vote_weighted.shape == (1, 1, 8, 2, 2)
    assert torch.allclose(pred_max, torch.full_like(pred_max, 20.0))
    assert torch.allclose(pred_mean, torch.full_like(pred_mean, 15.0))
    assert torch.allclose(pred_weighted, torch.full_like(pred_weighted, 94.0 / 6.0))


def test_invalid_final_decision_is_rejected(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported final_decision"):
        Bilateral_voting(
            torch.ones((1, 1, 8, 2, 2)),
            _identity_translation(2),
            _identity_translation(2),
            "median",
        )


def test_loss_factory_propagates_decision_and_rejects_nonstandard_layout(monkeypatch):
    _patch_cuda_to_noop(monkeypatch)

    args = _loss_args(final_decision="mean")
    loss_func, _, _, _, _ = build_loss_functions(
        args=args,
        hori_translation=_identity_translation(2),
        verti_translation=_identity_translation(2),
        label_mode="binary",
        fusion_enabled=False,
    )
    assert loss_func.final_decision == "mean"

    with pytest.raises(ValueError, match="standard8"):
        build_loss_functions(
            args=_loss_args(final_decision="weighted_mean", conn_layout="out8"),
            hori_translation=_identity_translation(2),
            verti_translation=_identity_translation(2),
            label_mode="binary",
            fusion_enabled=False,
        )

    fused, _, outer, _, _ = build_loss_functions(
        args=_loss_args(final_decision="mean"),
        hori_translation=_identity_translation(2),
        verti_translation=_identity_translation(2),
        label_mode="binary",
        fusion_enabled=True,
        fusion_branches="io",
    )
    assert fused.final_decision == "mean"
    assert outer.final_decision == "max"


def test_launcher_names_commands_and_schedules_include_final_decision(tmp_path):
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode="binary",
        dist_aux_loss="smooth_l1",
        final_decision="max",
    ) == "binary_8_bce"
    assert launcher.build_experiment_output_name(
        conn_num=8,
        label_mode="binary",
        dist_aux_loss="smooth_l1",
        final_decision="weighted_mean",
    ) == "binary_8_bce_fd_weighted_mean"
    with pytest.raises(ValueError, match="standard8"):
        launcher.build_experiment_output_name(
            conn_num=8,
            conn_layout="out8",
            label_mode="binary",
            dist_aux_loss="smooth_l1",
            final_decision="mean",
        )

    cmd = launcher.build_train_cmd(
        pybin="python",
        repo_root=tmp_path,
        preset={"dataset": "drive", "data_root": "data/DRIVE", "resize": (8, 8), "batch_size": 1},
        conn_num=8,
        conn_layout="standard8",
        final_decision="mean",
        label_mode="binary",
        dist_aux_loss="smooth_l1",
        dist_sf_l1_gamma=1.0,
        device=0,
        epochs=1,
        folds=1,
        target_fold=1,
    )
    assert cmd[cmd.index("--final_decision") + 1] == "mean"

    single_runs = launcher.build_single_schedule(
        {
            "dataset": "drive",
            "mode": "single",
            "single": {
                "conn_num": 8,
                "conn_layout": "standard8",
                "final_decision": "mean",
            },
            "output_dir": str(tmp_path),
        },
        device=0,
    )
    assert single_runs[0]["final_decision"] == "mean"
    assert launcher.get_run_experiment_name(single_runs[0]).endswith("_fd_mean")

    multi_runs = launcher.build_multi_schedule(
        {
            "dataset": ["drive"],
            "mode": "multi",
            "multi": {
                "conn_nums": [8],
                "conn_layouts": ["standard8"],
                "label_modes": ["binary"],
                "final_decisions": ["max", "mean"],
                "conn_fusions": ["none"],
            },
        },
        device=0,
        datasets=["drive"],
    )
    assert [run["final_decision"] for run in multi_runs] == ["max", "mean"]
    assert {
        launcher.get_run_experiment_name(run)
        for run in multi_runs
    } == {"binary_8_bce", "binary_8_bce_fd_mean"}
