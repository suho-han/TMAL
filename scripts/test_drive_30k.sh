#!/usr/bin/env bash
# Test segdiff and medsegdiff on DRIVE using epoch-30000 checkpoints.
# Runs both models in parallel: segdiff on GPU 0, medsegdiff on GPU 1.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

SEGDIFF_CKPT="${REPO_ROOT}/output_mnt/segdiff/drive/models/30000_model.pth"
MEDSEGDIFF_CKPT="${REPO_ROOT}/output_mnt/medsegdiff/drive/models/30000_model.pth"
SEGDIFF_LOG="${REPO_ROOT}/output_mnt/segdiff/drive/test_30k.log"
MEDSEGDIFF_LOG="${REPO_ROOT}/output_mnt/medsegdiff/drive/test_30k.log"

SEGDIFF_GPU="${1:-0}"
MEDSEGDIFF_GPU="${2:-1}"

for ckpt in "${SEGDIFF_CKPT}" "${MEDSEGDIFF_CKPT}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] Checkpoint not found: ${ckpt}" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --dataset drive
  --data_root data/DRIVE
  --resize 512 512
  --num-class 1
  --batch-size 1
  --folds 1
  --conn_num 8
  --conn_layout standard8
  --label_mode binary
  --dist_aux_loss smooth_l1
  --dist_sf_l1_gamma 1.0
  --conn_fusion none
  --fusion_branches io
  --fusion_lambda_fused 0.3
  --fusion_residual_scale 0.2
  --num_channels 3
  --diff_timesteps 100
  --diff_objective pred_noise
  --beta_schedule linear
  --output_dir output_mnt
  --test_only
)

cd "${REPO_ROOT}"

echo "[segdiff]    GPU ${SEGDIFF_GPU} | ckpt: 30000_model.pth | log: ${SEGDIFF_LOG}"
"${PYTHON}" src/engine/train_ablation.py \
  "${COMMON_ARGS[@]}" \
  --model segdiff \
  --experiment_name_suffix segdiff \
  --pretrained "${SEGDIFF_CKPT}" \
  --device "${SEGDIFF_GPU}" \
  >"${SEGDIFF_LOG}" 2>&1 &
SEGDIFF_PID=$!

echo "[medsegdiff] GPU ${MEDSEGDIFF_GPU} | ckpt: 30000_model.pth | log: ${MEDSEGDIFF_LOG}"
"${PYTHON}" src/engine/train_ablation.py \
  "${COMMON_ARGS[@]}" \
  --model medsegdiff \
  --experiment_name_suffix medsegdiff \
  --image_size 512 \
  --pretrained "${MEDSEGDIFF_CKPT}" \
  --device "${MEDSEGDIFF_GPU}" \
  >"${MEDSEGDIFF_LOG}" 2>&1 &
MEDSEGDIFF_PID=$!

echo ""
echo "segdiff    PID=${SEGDIFF_PID}"
echo "medsegdiff PID=${MEDSEGDIFF_PID}"
echo ""
echo "tail -f ${SEGDIFF_LOG}"
echo "tail -f ${MEDSEGDIFF_LOG}"
echo ""

wait "${SEGDIFF_PID}"
SEGDIFF_RC=$?
wait "${MEDSEGDIFF_PID}"
MEDSEGDIFF_RC=$?

echo ""
if ((SEGDIFF_RC == 0)); then
  echo "[segdiff]    DONE  -> $(ls "${REPO_ROOT}/output_mnt/segdiff/drive/final_results"*.csv 2>/dev/null | head -1)"
else
  echo "[segdiff]    FAILED (rc=${SEGDIFF_RC}) -> see ${SEGDIFF_LOG}"
fi
if ((MEDSEGDIFF_RC == 0)); then
  echo "[medsegdiff] DONE  -> $(ls "${REPO_ROOT}/output_mnt/medsegdiff/drive/final_results"*.csv 2>/dev/null | head -1)"
else
  echo "[medsegdiff] FAILED (rc=${MEDSEGDIFF_RC}) -> see ${MEDSEGDIFF_LOG}"
fi

exit $(( SEGDIFF_RC | MEDSEGDIFF_RC ))
