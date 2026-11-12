import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = binary_erosion(mask, border_value=0)
    return mask & ~eroded


def hd95(pred: np.ndarray, gt: np.ndarray, percentile: float = 95.0) -> float:
    pred_bin = np.asarray(pred) > 0.5
    gt_bin = np.asarray(gt) > 0.5

    if not pred_bin.any() or not gt_bin.any():
        return float("nan")

    pred_surface = _surface(pred_bin)
    gt_surface = _surface(gt_bin)

    if not pred_surface.any() or not gt_surface.any():
        return float("nan")

    dt_gt = distance_transform_edt(~gt_bin)
    dt_pred = distance_transform_edt(~pred_bin)

    d_pred_to_gt = dt_gt[pred_surface]
    d_gt_to_pred = dt_pred[gt_surface]

    return float(max(
        np.percentile(d_pred_to_gt, percentile),
        np.percentile(d_gt_to_pred, percentile),
    ))
