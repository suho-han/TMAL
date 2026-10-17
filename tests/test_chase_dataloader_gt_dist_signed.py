import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.GetDataset_CHASE import MyDataset_CHASE


class CHASEWithSignedDist(Dataset):
    """Test wrapper that augments MyDataset_CHASE with signed distance map loading."""

    def __init__(self, args, train_root, pat_ls, mode="test"):
        self.base = MyDataset_CHASE(args=args, train_root=train_root, pat_ls=pat_ls, mode=mode)
        self.dist_dir = Path(train_root) / "dist"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        img = sample["image"]
        mask = sample["label"]
        name = sample["name"]
        # name is like "1L" or "1R" from MyDataset_CHASE
        dist_path = self.dist_dir / f"Image_{name}_1stHO_dist.npy"
        dist = torch.from_numpy(np.load(dist_path)).float()
        return img, mask, dist, name


def _write_chase_mock_sample(root: Path, pat_id: int = 1, side: str = "L"):
    images_dir = root / "images"
    gt_dir = root / "gt"
    dist_dir = root / "dist"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    image_name = f"Image_{pat_id}{side}.jpg"
    gt_name = f"Image_{pat_id}{side}_1stHO.png"
    dist_name = f"Image_{pat_id}{side}_1stHO_dist.npy"

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 1] = 120

    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:24, 12:20] = 255

    signed_dist = np.linspace(-3.0, 3.0, 32 * 32, dtype=np.float32).reshape(32, 32)

    assert cv2.imwrite(str(images_dir / image_name), image)
    assert cv2.imwrite(str(gt_dir / gt_name), gt)
    np.save(dist_dir / dist_name, signed_dist)


def test_chase_dataloader_with_dist(tmp_path):
    # MyDataset_CHASE loads both L/R files per patient id.
    _write_chase_mock_sample(tmp_path, pat_id=1, side="L")
    _write_chase_mock_sample(tmp_path, pat_id=1, side="R")

    args = SimpleNamespace(resize=[960, 960])
    dataset = CHASEWithSignedDist(args=args, train_root=str(tmp_path), pat_ls=[1], mode="test")

    assert len(dataset) == 2

    img, mask, dist, name = dataset[0]
    assert img.shape == (3, 960, 960)
    assert mask.shape == (1, 960, 960)
    assert dist.shape == (32, 32)
    assert name in {"1L", "1R"}

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch_img, batch_mask, batch_dist, batch_name = next(iter(loader))

    assert batch_img.shape == (2, 3, 960, 960)
    assert batch_mask.shape == (2, 1, 960, 960)
    assert batch_dist.shape == (2, 32, 32)
    assert set(batch_name) == {"1L", "1R"}
