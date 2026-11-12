#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

usage() {
  echo "usage: $(basename "$0") <drive|deepcrack> <train|test> [extra train.py flags]" >&2
  echo "  e.g. $(basename "$0") drive train" >&2
  echo "       $(basename "$0") deepcrack test --device 1" >&2
  echo "       $(basename "$0") drive test --dry_run" >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

DATASET="$1"
MODE="$2"
shift 2

case "${DATASET}" in
  drive)
    CONFIG="${SCRIPT_DIR}/configs/drive_tmal_test.yaml"
    CKPT="${REPO_ROOT}/ckpts/drive_model.pth"
    ;;
  deepcrack)
    CONFIG="${SCRIPT_DIR}/configs/deepcrack_tmal_test.yaml"
    CKPT="${REPO_ROOT}/ckpts/deepcrack_model.pth"
    ;;
  *)
    echo "Unknown dataset: ${DATASET} (expected drive|deepcrack)" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" != "train" && "${MODE}" != "test" ]]; then
  usage
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing local virtualenv Python: ${PYTHON_BIN}" >&2
  exit 127
fi

ARGS=(--config "${CONFIG}" --mode "${MODE}")
if [[ "${MODE}" == "test" ]]; then
  ARGS+=(--pretrained "${CKPT}")
fi

# `--dry_run` is a run_from_config.py-level flag (print the train.py command
# without running it), not a train.py flag, so pull it out of the pass-through
# args before the `--` separator.
REMAINING=()
for ARG in "$@"; do
  if [[ "${ARG}" == "--dry_run" ]]; then
    ARGS+=(--dry_run)
  else
    REMAINING+=("${ARG}")
  fi
done

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_from_config.py" "${ARGS[@]}" -- "${REMAINING[@]}"
