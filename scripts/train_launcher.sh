#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
HELPER_SCRIPT="${SCRIPT_DIR}/train_launcher_from_config.py"

if [[ ! -f "${HELPER_SCRIPT}" ]]; then
  echo "Missing launcher helper: ${HELPER_SCRIPT}" >&2
  exit 127
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python interpreter: ${PYTHON_BIN}" >&2
  echo "Create .venv first (used by launcher for all datasets including RETOUCH)." >&2
  exit 127
fi

# Supported flags (forwarded to python helper):
#   --config <yaml> [--device N] [--batch-size N] [--dry_run] [--smoke] [--smoke_limit N]
#   final_decision is configured in YAML: max, mean, or weighted_mean
# `--smoke` runs all generated experiments by default; use `--smoke_limit` to trim.

CONFIG_PATH=""
PREV=""
for ARG in "$@"; do
  if [[ "${PREV}" == "--config" ]]; then
    CONFIG_PATH="${ARG}"
    break
  fi
  if [[ "${ARG}" == --config=* ]]; then
    CONFIG_PATH="${ARG#--config=}"
    break
  fi
  PREV="${ARG}"
done

if [[ -n "${CONFIG_PATH}" ]]; then
  RESOLVED_CONFIG="${CONFIG_PATH}"
  if [[ "${RESOLVED_CONFIG}" != /* && ! -f "${RESOLVED_CONFIG}" ]]; then
    ALT_CONFIG="${REPO_ROOT}/${RESOLVED_CONFIG}"
    if [[ -f "${ALT_CONFIG}" ]]; then
      RESOLVED_CONFIG="${ALT_CONFIG}"
    fi
  fi
  if [[ ! -f "${RESOLVED_CONFIG}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}" >&2
    exit 2
  fi
fi

exec "${PYTHON_BIN}" "${HELPER_SCRIPT}" "$@"
