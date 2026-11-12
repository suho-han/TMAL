import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aggregate_results import aggregate_root, parse_final_summary_csv  # noqa: E402


def _write_binary_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array.astype(np.uint8) * 255), mode="L").save(path)


def _write_missing_hd95_result(root: Path) -> None:
    (root / "models" / "checkpoint_batches" / "epoch_001" / "pred").mkdir(
        parents=True, exist_ok=True
    )
    (root / "models" / "checkpoint_batches" / "epoch_001" / "mask").mkdir(
        parents=True, exist_ok=True
    )

    pred = np.zeros((8, 8), dtype=np.uint8)
    pred[2:6, 2:6] = 1
    mask = pred.copy()

    _write_binary_png(
        root / "models" / "checkpoint_batches" / "epoch_001" / "pred" / "batch_0000.png",
        pred,
    )
    _write_binary_png(
        root / "models" / "checkpoint_batches" / "epoch_001" / "mask" / "batch_0000.png",
        mask,
    )

    with (root / "final_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "result_type",
                "evaluated_split",
                "eval_epoch",
                "checkpoint",
                "train_loss",
                "eval_loss",
                "dice",
                "jac",
                "cldice",
                "precision",
                "accuracy",
                "betti_error_0",
                "betti_error_1",
                "elapsed_hms",
                "hd95",
            ]
        )
        writer.writerow([
            "final",
            "test",
            "1",
            "best_model.pth",
            "0.100000",
            "0.200000",
            "0.900000",
            "0.800000",
            "0.700000",
            "nan",
            "nan",
            "0.000000",
            "0.000000",
            "00:00:01",
            "nan",
        ])


def test_aggregate_root_recovers_missing_hd95_from_checkpoint_batches(tmp_path):
    root = Path(tmp_path) / "drive"
    _write_missing_hd95_result(root)

    fold_rows, mean_row, std_row = aggregate_root(str(root), ["1"], "final_results.csv")

    assert len(fold_rows) == 1
    assert fold_rows[0]["best_hd95"] == 0.0
    assert mean_row["best_hd95"] == 0.0
    assert std_row["best_hd95"] == 0.0


def test_final_summary_csv_recovers_missing_hd95_from_checkpoint_batches(tmp_path):
    root = Path(tmp_path) / "external"
    _write_missing_hd95_result(root)

    metrics = parse_final_summary_csv(str(root / "final_results.csv"))

    assert metrics is not None
    assert metrics["best_hd95"] == 0.0