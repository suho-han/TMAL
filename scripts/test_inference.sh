#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
CHECKPOINT="${REPO_ROOT}/ckpts/drive_model.pth"
DATA_ROOT="${REPO_ROOT}/data/DRIVE_smoke"
OUTPUT_ROOT="${REPO_ROOT}/output_smoke"
EXPERIMENT_NAME="dist_inverted_dg_dsf_io_8_smooth_l1_segaux_w0.5_c3-0.5_c5-0.3"
EXPERIMENT_DIR="${OUTPUT_ROOT}/drive/${EXPERIMENT_NAME}"
LOG_PATH="${EXPERIMENT_DIR}/inference_smoke.log"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing local virtualenv Python: ${PYTHON_BIN}" >&2
  echo "Create it with: uv venv --python 3.12 .venv" >&2
  exit 127
fi
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

# The checkout has no DRIVE images. Build one tiny DRIVE-format train/test pair
# so the normal dataset loader, checkpoint loader, model, voting, metrics, and
# prediction writer all execute end to end.
if [[ ! -s "${DATA_ROOT}/test/images/smoke_test.tif" ]]; then
  "${PYTHON_BIN}" - "${DATA_ROOT}" <<'PY'
import os
import sys

import numpy as np
from PIL import Image

root = sys.argv[1]
rng = np.random.default_rng(42)
for split, stem, phase in (('train', 'smoke_train', 0.0), ('test', 'smoke_test', 0.35)):
    image_dir = os.path.join(root, split, 'images')
    mask_dir = os.path.join(root, split, 'masks')
    dist_dir = os.path.join(root, split, 'masks_dist_inverted')
    for directory in (image_dir, mask_dir, dist_dir):
        os.makedirs(directory, exist_ok=True)

    height, width = 32, 32
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width]
    center_x = 16 + int(3 * np.sin(phase))
    mask = (((yy - 16) ** 2 + (xx - center_x) ** 2) <= 6 ** 2).astype(np.float32)
    mask[((xx - (4 + int(phase * 10))) ** 2 + (yy - 22) ** 2) <= 2 ** 2] = 1.0

    Image.fromarray(image, mode='RGB').save(
        os.path.join(image_dir, f'{stem}.tif'), format='TIFF'
    )
    Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(
        os.path.join(mask_dir, f'{stem}.gif'), format='GIF'
    )
    np.save(os.path.join(dist_dir, f'{stem}_dist_inverted.npy'), mask)
PY
fi

cd "${REPO_ROOT}"
SECONDS=0
mkdir -p "${EXPERIMENT_DIR}"
echo "Running real CPU inference with ${CHECKPOINT}"
"${PYTHON_BIN}" src/engine/train.py \
  --device 0 \
  --dataset drive \
  --data_root "${DATA_ROOT}" \
  --resize 512 512 \
  --model dconnnet \
  --num_channels 3 \
  --num-class 1 \
  --batch-size 1 \
  --num_workers 0 \
  --epochs 500 \
  --folds 1 \
  --conn_num 8 \
  --conn_layout standard8 \
  --label_mode dist_inverted \
  --dist_aux_loss smooth_l1 \
  --dist_sf_l1_gamma 1.0 \
  --conn_fusion dg_dsf \
  --fusion_branches io \
  --fusion_lambda_middle 0.05 \
  --fusion_lambda_fused 0.3 \
  --fusion_residual_scale 0.2 \
  --use_seg_aux \
  --seg_aux_weight 0.5 \
  --fusion_gate_reg_weight 0.01 \
  --conn_aux_c3_weight 0.5 \
  --conn_aux_c4_weight 0.2 \
  --conn_aux_c5_weight 0.3 \
  --conn_aux_c6_weight 0.2 \
  --conn_aux_c7_weight 0.2 \
  --pretrained "${CHECKPOINT}" \
  --test_only \
  --max_eval_batches 1 \
  --output_dir "${OUTPUT_ROOT}" \
  --experiment_name_suffix c3-0.5_c5-0.3 \
  --save-per-epochs 50 \
  2>&1 | tee "${LOG_PATH}"

RESULTS_PATH="${EXPERIMENT_DIR}/final_results.csv"
SAMPLES_PATH="${EXPERIMENT_DIR}/models/test_sample_metrics.csv"
PREDICTION_PATH="${EXPERIMENT_DIR}/models/checkpoint_batches/epoch_001/pred_samples/smoke_test.png"
test -s "${RESULTS_PATH}"
test -s "${SAMPLES_PATH}"
test -s "${PREDICTION_PATH}"

echo "SMOKE_SUCCESS=1"
echo "SMOKE_ELAPSED_SECONDS=${SECONDS}"
echo "SMOKE_RESULTS=${RESULTS_PATH}"
echo "SMOKE_PREDICTION=${PREDICTION_PATH}"
echo "--- final_results.csv ---"
tail -n 3 "${RESULTS_PATH}"
echo "--- test_sample_metrics.csv ---"
tail -n 3 "${SAMPLES_PATH}"
