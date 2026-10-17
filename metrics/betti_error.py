import numpy as np
import torch
from skimage import measure


def _mask_to_binary_numpy(mask):
    if torch.is_tensor(mask):
        binary = mask.detach().cpu().numpy()
    else:
        binary = np.asarray(mask)

    binary = np.squeeze(binary)
    if binary.ndim != 2:
        raise ValueError(
            f'Betti metrics expect a 2D mask after squeeze, got shape {binary.shape}'
        )

    return (binary > 0).astype(np.uint8)


def betti_numbers(mask):
    binary = _mask_to_binary_numpy(mask)
    betti_0 = int(measure.label(binary, connectivity=1).max())
    euler = int(measure.euler_number(binary, connectivity=1))
    betti_1 = max(0, betti_0 - euler)
    return betti_0, betti_1


def betti_error(pred_mask, target_mask):
    pred_betti_0, pred_betti_1 = betti_numbers(pred_mask)
    target_betti_0, target_betti_1 = betti_numbers(target_mask)

    return {
        'pred_betti_0': pred_betti_0,
        'pred_betti_1': pred_betti_1,
        'target_betti_0': target_betti_0,
        'target_betti_1': target_betti_1,
        'betti_error_0': abs(pred_betti_0 - target_betti_0),
        'betti_error_1': abs(pred_betti_1 - target_betti_1),
    }
