import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_datasets
from scripts import train_launcher_from_config as launcher


def _write_sample(root: Path, mode: str, stem: str) -> None:
    img_dir = root / f"{mode}_img"
    lab_dir = root / f"{mode}_lab"
    dist_dir = root / f"{mode}_lab_dist"
    dist_inv_dir = root / f"{mode}_lab_dist_inverted"
    for d in (img_dir, lab_dir, dist_dir, dist_inv_dir):
        d.mkdir(parents=True, exist_ok=True)

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = (128, 64, 32)
    cv2.imwrite(str(img_dir / f"{stem}.jpg"), image)

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(lab_dir / f"{stem}.png"), mask)

    dist = np.linspace(0.0, 1.0, num=32 * 32, dtype=np.float32).reshape(32, 32)
    np.save(dist_dir / f"{stem}_dist.npy", dist)
    np.save(dist_inv_dir / f"{stem}_dist_inverted.npy", 1.0 - dist)


def _make_deepcrack_root(tmp_path: Path) -> Path:
    root = tmp_path / "DeepCrack"
    for stem in ("11111", "11112"):
        _write_sample(root, "train", stem)
    _write_sample(root, "test", "11304")
    return root


def _make_args(data_root: Path, label_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="deepcrack",
        data_root=str(data_root),
        resize=[64, 64],
        label_mode=label_mode,
        batch_size=1,
    )


@pytest.mark.parametrize("label_mode", ["binary", "dist", "dist_inverted"])
def test_build_datasets_deepcrack_returns_expected_dict_format(tmp_path, label_mode):
    data_root = _make_deepcrack_root(tmp_path)
    args = _make_args(data_root=data_root, label_mode=label_mode)

    trainset, validset, testset = build_datasets(args)

    assert validset is None
    assert len(trainset) == 2
    assert len(testset) == 1

    sample = testset[0]
    assert set(sample.keys()) == {"image", "label", "binary_gt", "name"}
    assert tuple(sample["image"].shape) == (3, 64, 64)
    assert tuple(sample["label"].shape) == (1, 64, 64)
    assert tuple(sample["binary_gt"].shape) == (1, 64, 64)
    assert sample["name"] == "11304"

    binary_unique = set(np.unique(sample["binary_gt"].numpy()).tolist())
    assert binary_unique.issubset({0.0, 1.0})

    if label_mode == "binary":
        label_unique = set(np.unique(sample["label"].numpy()).tolist())
        assert label_unique.issubset({0.0, 1.0})
    else:
        assert np.isfinite(sample["label"].numpy()).all()


def test_deepcrack_train_augmentation_does_not_crash(tmp_path):
    data_root = _make_deepcrack_root(tmp_path)
    args = _make_args(data_root=data_root, label_mode="binary")

    trainset, _, _ = build_datasets(args)
    sample = trainset[0]
    assert tuple(sample["image"].shape) == (3, 64, 64)


def test_launcher_deepcrack_preset():
    config = {
        "mode": "single",
        "dataset": "deepcrack",
        "device": 0,
        "single": {
            "conn_num": 8,
            "label_mode": "binary",
            "output_dir": "output",
        },
    }
    runs = launcher.build_single_schedule(config, device=0)
    assert len(runs) == 1
    preset = runs[0]["preset"]
    assert preset["dataset"] == "deepcrack"
    assert preset["data_root"] == "data/DeepCrack"
    assert tuple(preset["resize"]) == (512, 512)
