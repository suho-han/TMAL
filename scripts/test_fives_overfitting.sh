#!/usr/bin/env bash
# Run test-only eval for epochs 100, 200, 300 and send results to Discord.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

EXP="dist_inverted_dg_direct_io_8_smooth_l1_segaux_w0.3_c5-0.2"
EXP_DIR="output/fives/${EXP}"
LOG_FILE="${EXP_DIR}/test_overfitting.log"

COMMON=(
  --dataset fives --data_root data/FIVES
  --resize 512 512 --num-class 1 --batch-size 8 --epochs 500
  --lr 0.0038 --lr-update poly --optimizer adam --weight_decay 0.0 --folds 1
  --conn_num 8 --conn_layout standard8 --label_mode dist_inverted
  --dist_aux_loss smooth_l1 --conn_fusion dg_direct --fusion_branches io
  --fusion_lambda_middle 0.05 --fusion_lambda_fused 0.3 --fusion_residual_scale 0.2
  --device 0 --model dconnnet --use_seg_aux --seg_aux_weight 0.3
  --fusion_gate_reg_weight 0.01 --conn_aux_c3_weight 0.3 --conn_aux_c5_weight 0.2
  --output_dir ""
)

echo "[$(date '+%H:%M:%S')] Starting overfitting test for ${EXP}" | tee "$LOG_FILE"

for EPOCH in 100 200 300; do
  SAVE="${EXP_DIR}/test_e${EPOCH}"
  CKPT="${EXP_DIR}/models/${EPOCH}_model.pth"

  # Skip if already complete
  if [ -f "${SAVE}/final_results.csv" ]; then
    echo "[$(date '+%H:%M:%S')] Epoch ${EPOCH}: already done, skipping." | tee -a "$LOG_FILE"
    continue
  fi

  echo "[$(date '+%H:%M:%S')] Epoch ${EPOCH}: starting..." | tee -a "$LOG_FILE"
  rm -rf "${SAVE}"

  uv run src/engine/train.py \
    "${COMMON[@]}" \
    --save "$SAVE" \
    --test_only \
    --pretrained "$CKPT" \
    >> "$LOG_FILE" 2>&1

  if [ -f "${SAVE}/final_results.csv" ]; then
    echo "[$(date '+%H:%M:%S')] Epoch ${EPOCH}: DONE" | tee -a "$LOG_FILE"
  else
    echo "[$(date '+%H:%M:%S')] Epoch ${EPOCH}: FAILED (no final_results.csv)" | tee -a "$LOG_FILE"
  fi
done

echo "[$(date '+%H:%M:%S')] All epochs done. Collecting results..." | tee -a "$LOG_FILE"

# Parse results and get train losses
RESULTS=$(python3 - <<'PYEOF'
import csv, sys

exp_dir = "output/fives/dist_inverted_dg_direct_io_8_smooth_l1_segaux_w0.3_c5-0.2"

# Train losses
train_losses = {}
with open(f"{exp_dir}/results.csv") as f:
    for row in csv.DictReader(f):
        ep = row["epoch"].lstrip("0") or "0"
        train_losses[ep] = float(row["train_loss"])

lines = []
lines.append("**FIVES Overfitting Check** | `dist_inverted_dg_direct_io_8_smooth_l1_segaux_w0.3_c5-0.2`")
lines.append("```")
lines.append(f"{'Epoch':>6} | {'Train Loss':>10} | {'Test Loss':>9} | {'Dice':>6} | {'Jac':>6} | {'clDice':>7} | {'HD95':>7} | {'Betti0':>7} | {'Betti1':>7}")
lines.append("-" * 88)

for epoch in [100, 200, 300]:
    csv_path = f"{exp_dir}/test_e{epoch}/final_results.csv"
    try:
        with open(csv_path) as f:
            row = list(csv.DictReader(f))[0]
        tl = train_losses.get(str(epoch), float("nan"))
        el = float(row["eval_loss"])
        dice = float(row["dice"])
        jac = float(row["jac"])
        cldice = float(row["cldice"])
        hd95 = float(row["hd95"])
        b0 = float(row["betti_error_0"])
        b1 = float(row["betti_error_1"])
        lines.append(f"{epoch:>6} | {tl:>10.4f} | {el:>9.4f} | {dice:>6.4f} | {jac:>6.4f} | {cldice:>7.4f} | {hd95:>7.2f} | {b0:>7.2f} | {b1:>7.2f}")
    except Exception as e:
        lines.append(f"{epoch:>6} | {'N/A':>10} | ERROR: {e}")

lines.append("```")

# Overfitting judgment
try:
    results = {}
    for epoch in [100, 200, 300]:
        csv_path = f"{exp_dir}/test_e{epoch}/final_results.csv"
        with open(csv_path) as f:
            row = list(csv.DictReader(f))[0]
        results[epoch] = {
            "train_loss": train_losses.get(str(epoch), float("nan")),
            "test_loss": float(row["eval_loss"]),
            "dice": float(row["dice"]),
        }

    dice_vals = [results[e]["dice"] for e in [100, 200, 300]]
    test_losses = [results[e]["test_loss"] for e in [100, 200, 300]]
    train_losses_sel = [results[e]["train_loss"] for e in [100, 200, 300]]

    dice_trend = "상승" if dice_vals[2] > dice_vals[0] else ("하락" if dice_vals[2] < dice_vals[0] else "유지")
    test_loss_trend = "감소" if test_losses[2] < test_losses[0] else ("증가" if test_losses[2] > test_losses[0] else "유지")
    train_loss_trend = "감소" if train_losses_sel[2] < train_losses_sel[0] else ("증가" if train_losses_sel[2] > train_losses_sel[0] else "유지")

    if test_losses[2] > test_losses[0] and train_losses_sel[2] < train_losses_sel[0]:
        verdict = "**Overfitting 의심** — train loss 감소 중 test loss 증가"
    elif dice_vals[1] >= dice_vals[0] and dice_vals[2] >= dice_vals[1]:
        verdict = "**Overfitting 없음** — Dice 지속 향상"
    elif abs(dice_vals[2] - dice_vals[0]) < 0.005:
        verdict = "**수렴** — Dice 변화 미미 (<0.5%)"
    else:
        verdict = f"**추가 분석 필요** — Dice {dice_trend}, Test loss {test_loss_trend}"

    lines.append(f"Train loss: e100={train_losses_sel[0]:.4f} → e200={train_losses_sel[1]:.4f} → e300={train_losses_sel[2]:.4f} ({train_loss_trend})")
    lines.append(f"Test  Dice: e100={dice_vals[0]:.4f} → e200={dice_vals[1]:.4f} → e300={dice_vals[2]:.4f} ({dice_trend})")
    lines.append(f"판정: {verdict}")
except Exception as e:
    lines.append(f"판정 실패: {e}")

print("\n".join(lines))
PYEOF
)

echo "$RESULTS" | tee -a "$LOG_FILE"

# Send to Discord
uv run scripts/discord_alert.py \
  --job "FIVES overfitting check (e100/200/300)" \
  --message "$RESULTS" \
  --status "DONE"

echo "[$(date '+%H:%M:%S')] Discord notification sent." | tee -a "$LOG_FILE"
