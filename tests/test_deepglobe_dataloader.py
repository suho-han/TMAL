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


def _write_sample(split_dir: Path, stem: str) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = 32
    image[..., 1] = 128
    image[..., 2] = 224
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255

    cv2.imwrite(str(split_dir / f"{stem}_sat.jpg"), image)
    cv2.imwrite(str(split_dir / f"{stem}_mask.png"), mask)

    dist_dir = split_dir / "labels_dist"
    dist_inv_dir = split_dir / "labels_dist_inverted"
    dist_dir.mkdir(parents=True, exist_ok=True)
    dist_inv_dir.mkdir(parents=True, exist_ok=True)

    dist = np.linspace(0.0, 1.0, num=32 * 32, dtype=np.float32).reshape(32, 32)
    dist_inv = 1.0 - dist
    np.save(dist_dir / f"{stem}_mask_dist.npy", dist)
    np.save(dist_inv_dir / f"{stem}_mask_dist_inverted.npy", dist_inv)


def _make_deepglobe_splitted_root(tmp_path: Path) -> Path:
    root = tmp_path / "DeepGlobe" / "splitted"
    for split, stem in (("train", "100001"), ("val", "200001"), ("test", "300001")):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_sample(split_dir, stem=stem)
    return root


def _make_args(data_root: Path, label_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="deepglobe",
        data_root=str(data_root),
        resize=[64, 64],
        label_mode=label_mode,
        batch_size=1,
    )


@pytest.mark.parametrize("label_mode", ["binary", "dist", "dist_inverted"])
def test_build_datasets_deepglobe_returns_expected_dict_format(tmp_path, label_mode):
    data_root = _make_deepglobe_splitted_root(tmp_path)
    args = _make_args(data_root=data_root, label_mode=label_mode)

    trainset, validset, testset = build_datasets(args)

    assert len(trainset) == 1
    assert len(validset) == 1
    assert len(testset) == 1

    sample = validset[0]
    assert set(sample.keys()) == {"image", "label", "binary_gt", "name"}
    assert tuple(sample["image"].shape) == (3, 64, 64)
    assert tuple(sample["label"].shape) == (1, 64, 64)
    assert tuple(sample["binary_gt"].shape) == (1, 64, 64)
    assert sample["name"] == "200001"

    binary_unique = set(np.unique(sample["binary_gt"].numpy()).tolist())
    assert binary_unique.issubset({0.0, 1.0})

    if label_mode == "binary":
        label_unique = set(np.unique(sample["label"].numpy()).tolist())
        assert label_unique.issubset({0.0, 1.0})
    else:
        assert np.isfinite(sample["label"].numpy()).all()


def test_launcher_deepglobe_preset_and_schedule():
    config = {
        "mode": "single",
        "dataset": "deepglobe",
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
    assert preset["dataset"] == "deepglobe"
    assert preset["data_root"] == "data/DeepGlobe/splitted"
    assert tuple(preset["resize"]) == (1024, 1024)
