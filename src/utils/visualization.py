import os

import torchvision.utils as tv_utils


def save_checkpoint_batch_triplet(save_dir, images, preds, masks, epoch, batch_idx, sample_names=None):
    """Save image/pred/mask triplet to disk for visual inspection.

    Saves two sets of files:
    - batch grids: <base>/image|pred|mask/batch_XXXX.png  (all samples in one grid)
    - per-sample:  <base>/pred_samples|mask_samples/<sample_name>.png
    """
    base_dir = os.path.join(save_dir, 'models', 'checkpoint_batches', f'epoch_{epoch:03d}')

    # Batch-level grid images (existing behavior, kept for visual inspection)
    for name, tensor in [('image', images), ('pred', preds), ('mask', masks)]:
        d = os.path.join(base_dir, name)
        os.makedirs(d, exist_ok=True)
        tv_utils.save_image(
            tensor.detach().cpu().float(),
            os.path.join(d, f'batch_{batch_idx:04d}.png'),
            normalize=True, scale_each=True,
        )

    # Per-sample binary images named by sample_name (enables offline metric computation)
    batch_size = preds.shape[0]
    for s_idx in range(batch_size):
        if sample_names is not None and s_idx < len(sample_names):
            sample_id = str(sample_names[s_idx]).replace('/', '_').replace('\\', '_')
        else:
            sample_id = f'b{batch_idx:04d}s{s_idx:02d}'
        for name, tensor in [('pred_samples', preds), ('mask_samples', masks)]:
            d = os.path.join(base_dir, name)
            os.makedirs(d, exist_ok=True)
            tv_utils.save_image(
                tensor[s_idx].detach().cpu().float(),
                os.path.join(d, f'{sample_id}.png'),
                normalize=False,
            )
