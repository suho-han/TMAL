#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/gpu_train_process_summary.sh [--all]

Options:
  --all   Include GPU compute processes that are not train.py.
  --short Print only the active-process table; skip schedule status.
  -h, --help
EOF
}

trim() {
  local value="$*"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

normalize_conn_fusion() {
  local conn_fusion="${1:-none}"
  if [[ -z "${conn_fusion}" ]]; then
    conn_fusion="none"
  fi
  if [[ "${conn_fusion}" == "decoder_guided" ]]; then
    conn_fusion="dg"
  fi
  printf '%s' "${conn_fusion}"
}

normalize_output_dir() {
  local output_dir="$1"
  if [[ -z "${output_dir}" ]]; then
    printf 'output'
    return 0
  fi
  if [[ "${output_dir}" == "/" ]]; then
    printf '/'
    return 0
  fi
  output_dir="${output_dir%/}"
  if [[ -z "${output_dir}" ]]; then
    printf '.'
    return 0
  fi
  printf '%s' "${output_dir}"
}

build_experiment_name() {
  local label_mode="$1"
  local conn_num="$2"
  local dist_aux_loss="$3"
  local conn_layout="$4"
  local conn_fusion="$5"
  local legacy_fusion_token="$6"
  local use_seg_aux="${7:-0}"
  local seg_aux_weight="${8:-}"
  local fusion_residual_scale="${9:-0.2}"
  local fusion_branches="${10:-none}"
  local base_name
  local layout_suffix=""

  if [[ -z "${conn_layout}" ]]; then
    if [[ "${conn_num}" == "24" ]]; then
      conn_layout="full24"
    else
      conn_layout="standard8"
    fi
  fi
  if [[ "${conn_num}" == "8" && "${conn_layout}" != "standard8" ]]; then
    layout_suffix="_${conn_layout}"
  fi

  conn_fusion="$(normalize_conn_fusion "${conn_fusion}")"
  # fusion profiles removed: ignore the legacy token and do not include
  # a profile token in experiment names. Keep parameter for backward compatibility.

  if [[ "${conn_fusion}" == "none" ]]; then
    if [[ "${label_mode}" == "binary" ]]; then
      base_name="binary_${conn_num}${layout_suffix}_bce"
    else
      base_name="${label_mode}_${conn_num}${layout_suffix}_${dist_aux_loss}"
    fi
  else
    local fusion_tag="${conn_fusion}"
    if [[ "${conn_fusion}" == "scaled_sum" ]]; then
      local scale_str
      scale_str="$(awk -v s="${fusion_residual_scale}" 'BEGIN { printf "%.6f", s + 0 }' | sed -E 's/0+$//; s/\.$//')"
      fusion_tag="${fusion_tag}_rs${scale_str}"
    fi
    local fb
    fb="$(trim "${fusion_branches}")"
    if [[ -n "${fb}" && "${fb}" != "none" ]]; then
      fusion_tag="${fusion_tag}_${fb}"
    fi
    if [[ "${label_mode}" == "binary" ]]; then
      base_name="binary_${fusion_tag}_${conn_num}${layout_suffix}_bce"
    else
      base_name="${label_mode}_${fusion_tag}_${conn_num}${layout_suffix}_${dist_aux_loss}"
    fi
  fi

  if [[ "${use_seg_aux}" == "1" ]]; then
    if [[ -n "${seg_aux_weight}" ]]; then
      base_name="${base_name}_segaux_w${seg_aux_weight}"
    else
      base_name="${base_name}_segaux"
    fi
  fi

  printf '%s' "${base_name}"
}

parse_experiment_name_fields() {
  local experiment_name="$1"
  local base_name
  local label_mode conn_num conn_layout dist_aux_loss conn_fusion

  base_name="${experiment_name}"

  # Strip SegAux suffixes so metadata parsing stays stable.
  # Examples:
  # - dist_inverted_8_smooth_l1_segaux
  # - dist_inverted_8_smooth_l1_segaux_w0.5
  if [[ "${base_name}" =~ ^(.+)_segaux_w([0-9.eE+-]+)$ ]]; then
    base_name="${BASH_REMATCH[1]}"
  elif [[ "${base_name}" =~ ^(.+)_segaux$ ]]; then
    base_name="${BASH_REMATCH[1]}"
  fi

  if [[ "${base_name}" =~ ^binary_([A-Za-z0-9_]+)_([A-Za-z0-9]+)_rs([0-9]*\.?[0-9]+)_([0-9]+)(_([A-Za-z0-9]+))?_bce$ ]]; then
    label_mode="binary"
    conn_fusion="${BASH_REMATCH[1]}"
    conn_fusion="$(normalize_conn_fusion "${conn_fusion}")"
    # legacy profile token captured by regex in group 2 is ignored
    conn_num="${BASH_REMATCH[4]}"
    conn_layout="${BASH_REMATCH[6]}"
    if [[ -z "${conn_layout}" ]]; then
      conn_layout="standard8"
    fi
    dist_aux_loss="smooth_l1"
    printf '%s\t%s\t%s\t%s\t%s' "${label_mode}" "${conn_num}" "${conn_layout}" "${dist_aux_loss}" "${conn_fusion}"
    return 0
  fi

  if [[ "${base_name}" =~ ^binary_([A-Za-z0-9_]+)_([A-Za-z0-9]+)_([0-9]+)(_([A-Za-z0-9]+))?_bce$ ]]; then
    label_mode="binary"
    conn_fusion="${BASH_REMATCH[1]}"
    conn_fusion="$(normalize_conn_fusion "${conn_fusion}")"
    conn_num="${BASH_REMATCH[3]}"
    conn_layout="${BASH_REMATCH[5]}"
    if [[ -z "${conn_layout}" ]]; then
      conn_layout="standard8"
    fi
    dist_aux_loss="smooth_l1"
    printf '%s\t%s\t%s\t%s\t%s' "${label_mode}" "${conn_num}" "${conn_layout}" "${dist_aux_loss}" "${conn_fusion}"
    return 0
  fi

  if [[ "${base_name}" =~ ^binary_([0-9]+)(_([A-Za-z0-9]+))?_bce$ ]]; then
    label_mode="binary"
    conn_fusion="none"
    conn_num="${BASH_REMATCH[1]}"
    conn_layout="${BASH_REMATCH[3]}"
    if [[ -z "${conn_layout}" ]]; then
      conn_layout="standard8"
    fi
    dist_aux_loss="smooth_l1"
    printf '%s\t%s\t%s\t%s\t%s' "${label_mode}" "${conn_num}" "${conn_layout}" "${dist_aux_loss}" "${conn_fusion}"
    return 0
  fi

  if [[ "${base_name}" =~ ^(dist|dist_inverted)_([A-Za-z0-9_]+)_([A-Za-z0-9]+)_rs([0-9]*\.?[0-9]+)_([0-9]+)(_([A-Za-z0-9]+))?_(.+)$ ]]; then
    label_mode="${BASH_REMATCH[1]}"
    conn_fusion="${BASH_REMATCH[2]}"
    conn_fusion="$(normalize_conn_fusion "${conn_fusion}")"
    conn_num="${BASH_REMATCH[5]}"
    conn_layout="${BASH_REMATCH[7]}"
    if [[ -z "${conn_layout}" ]]; then
      if [[ "${conn_num}" == "24" ]]; then
        conn_layout="full24"
      else
        conn_layout="standard8"
      fi
    fi
    dist_aux_loss="${BASH_REMATCH[8]}"
    printf '%s\t%s\t%s\t%s\t%s' "${label_mode}" "${conn_num}" "${conn_layout}" "${dist_aux_loss}" "${conn_fusion}"
    return 0
  fi

  if [[ "${base_name}" =~ ^(dist|dist_inverted)_([A-Za-z0-9_]+)_([A-Za-z0-9]+)_([0-9]+)(_([A-Za-z0-9]+))?_(.+)$ ]]; then
    label_mode="${BASH_REMATCH[1]}"
    conn_fusion="${BASH_REMATCH[2]}"
    conn_fusion="$(normalize_conn_fusion "${conn_fusion}")"
    conn_num="${BASH_REMATCH[4]}"
    conn_layout="${BASH_REMATCH[6]}"
    if [[ -z "${conn_layout}" ]]; then
      if [[ "${conn_num}" == "24" ]]; then
        conn_layout="full24"
      else
        conn_layout="standard8"
      fi
    fi
    dist_aux_loss="${BASH_REMATCH[7]}"
    printf '%s\t%s\t%s\t%s\t%s' "${label_mode}" "${conn_num}" "${conn_layout}" "${dist_aux_loss}" "${conn_fusion}"
    return 0
  fi

  if [[ "${base_name}" =~ ^(.+)_([0-9]+)(_([A-Za-z0-9]+))?_(.+)$ ]]; then
    label_mode="${BASH_REMATCH[1]}"
    conn_fusion="none"
    conn_num="${BASH_REMATCH[2]}"
    conn_layout="${BASH_REMATCH[4]}"
    if [[ -z "${conn_layout}" ]]; then
      if [[ "${conn_num}" == "24" ]]; then
        conn_layout="full24"
      else
        conn_layout="standard8"
      fi
    fi
    dist_aux_loss="${BASH_REMATCH[5]}"
    printf '%s\t%s\t%s\t%s\t%s' "${label_mode}" "${conn_num}" "${conn_layout}" "${dist_aux_loss}" "${conn_fusion}"
    return 0
  fi

  return 1
}

resolve_results_csv_path() {
  local output_base="$1"
  local dataset="$2"
  local experiment_name="$3"
  local suffix_dir="${4:-}"
  local output_base_name output_parent_name
  local -a candidates=()
  local -A seen=()
  local candidate first_candidate

  output_base_name="$(basename -- "${output_base}")"
  output_parent_name="$(basename -- "$(dirname -- "${output_base}")")"

  if [[ "${output_base_name}" == "${experiment_name}" ]]; then
    candidates+=("${output_base}/results.csv")
  fi
  if [[ "${output_base_name}" == "${dataset}" ]]; then
    candidates+=("${output_base}/${experiment_name}/results.csv")
  fi
  if [[ "${output_parent_name}" == "${dataset}" && "${output_base_name}" == "${experiment_name}" ]]; then
    candidates+=("${output_base}/results.csv")
  fi

  if [[ -n "${suffix_dir}" ]]; then
    candidates+=(
      "${output_base}/${suffix_dir}/${dataset}/results.csv"
      "${output_base}/${suffix_dir}/${dataset}/${experiment_name}/results.csv"
    )
  fi

  candidates+=(
    "${output_base}/${dataset}/${experiment_name}/results.csv"
    "${output_base}/${experiment_name}/results.csv"
    "${output_base}/results.csv"
  )

  first_candidate=""
  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -n "${seen["${candidate}"]+x}" ]]; then
      continue
    fi
    seen["${candidate}"]=1
    if [[ -z "${first_candidate}" ]]; then
      first_candidate="${candidate}"
    fi
    if [[ -r "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done

  printf '%s' "${first_candidate}"
}

format_duration_hms() {
  local total_seconds="$1"
  local days hours minutes seconds remainder
  if ((total_seconds < 0)); then
    total_seconds=0
  fi
  days=$((total_seconds / 86400))
  remainder=$((total_seconds % 86400))
  hours=$((remainder / 3600))
  remainder=$((remainder % 3600))
  minutes=$((remainder / 60))
  seconds=$((remainder % 60))
  if ((days > 0)); then
    printf '%dd %02d:%02d:%02d' "${days}" "${hours}" "${minutes}" "${seconds}"
  else
    printf '%02d:%02d:%02d' "${hours}" "${minutes}" "${seconds}"
  fi
}

format_finish_time_from_epoch() {
  local finish_epoch="$1"
  local formatted
  if ! formatted="$(date -d "@${finish_epoch}" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)"; then
    if ! formatted="$(date -r "${finish_epoch}" '+%Y-%m-%d %H:%M:%S %Z' 2>/dev/null)"; then
      formatted="n/a"
    fi
  fi
  printf '%s' "${formatted}"
}

compute_eta_fields() {
  local csv_path="$1"
  local total_epochs_raw="$2"
  local parsed rc marker last_epoch elapsed_sec valid_rows total_epochs remaining_epochs eta_sec predicted_total_sec now_epoch finish_epoch eta_duration eta_finish eta_status progress

  if [[ ! -r "${csv_path}" ]]; then
    printf 'n/a\tn/a\tn/a\tno_csv\tn/a\tn/a'
    return 0
  fi

  if ! parsed="$(awk -F',' '
    function trim(s) {
      gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s)
      return s
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        key = trim($i)
        idx[key] = i
      }
      if (!("epoch" in idx) || !("elapsed_hms" in idx)) {
        exit 10
      }
      epoch_idx = idx["epoch"]
      elapsed_idx = idx["elapsed_hms"]
      next
    }
    {
      epoch_raw = trim($(epoch_idx))
      elapsed_raw = trim($(elapsed_idx))
      if (epoch_raw == "" || elapsed_raw == "") {
        next
      }
      if (epoch_raw !~ /^[0-9]+$/) {
        next
      }
      count = split(elapsed_raw, t, ":")
      if (count != 3) {
        next
      }
      if (t[1] !~ /^[0-9]+$/ || t[2] !~ /^[0-9]+$/ || t[3] !~ /^[0-9]+$/) {
        next
      }
      if ((t[2] + 0) >= 60 || (t[3] + 0) >= 60) {
        next
      }

      epoch = epoch_raw + 0
      elapsed_sec = (t[1] * 3600) + (t[2] * 60) + t[3]
      valid_rows += 1

      if (epoch > best_epoch || (epoch == best_epoch && elapsed_sec > best_elapsed_sec)) {
        best_epoch = epoch
        best_elapsed_sec = elapsed_sec
      }
    }
    END {
      if (valid_rows == 0) {
        exit 11
      }
      printf "OK,%d,%d,%d\n", best_epoch, best_elapsed_sec, valid_rows
    }
  ' "${csv_path}" 2>/dev/null)"; then
    rc=$?
    case "${rc}" in
      10)
        printf 'n/a\tn/a\tn/a\tmissing_cols\tn/a\tn/a'
        ;;
      11)
        printf 'n/a\tn/a\tn/a\tno_valid_rows\tn/a\tn/a'
        ;;
      *)
        printf 'n/a\tn/a\tn/a\tparse_error\tn/a\tn/a'
        ;;
    esac
    return 0
  fi

  IFS=',' read -r marker last_epoch elapsed_sec valid_rows <<< "${parsed}"
  if [[ "${marker}" != "OK" ]] || [[ -z "${last_epoch}" ]] || [[ -z "${elapsed_sec}" ]]; then
    printf 'n/a\tn/a\tn/a\tparse_error\tn/a\tn/a'
    return 0
  fi
  if ((last_epoch <= 0)); then
    printf 'n/a\tn/a\tn/a\tinvalid_epoch\tn/a\tn/a'
    return 0
  fi

  total_epochs=0
  if [[ "${total_epochs_raw}" =~ ^[0-9]+$ ]] && ((total_epochs_raw > 0)); then
    total_epochs="${total_epochs_raw}"
  else
    total_epochs="${last_epoch}"
  fi

  remaining_epochs=$((total_epochs - last_epoch))
  if ((remaining_epochs > 0)); then
    eta_status="running"
  else
    eta_status="completed_or_overrun"
    remaining_epochs=0
  fi

  eta_sec="$(awk -v elapsed="${elapsed_sec}" -v epoch="${last_epoch}" -v remain="${remaining_epochs}" 'BEGIN { printf "%d", ((elapsed / epoch) * remain) + 0.5 }')"
  predicted_total_sec=$((elapsed_sec + eta_sec))
  eta_duration="$(format_duration_hms "${eta_sec}")"

  now_epoch="$(date +%s)"
  finish_epoch=$((now_epoch + eta_sec))
  eta_finish="$(format_finish_time_from_epoch "${finish_epoch}")"

  progress="${last_epoch}/${total_epochs}"
  printf '%s\t%s\t%s\t%s\t%s\t%s' "${progress}" "${eta_duration}" "${eta_finish}" "${eta_status}" "${eta_sec}" "${predicted_total_sec}"
}

extract_flag() {
  local key="$1"
  local default_value="$2"
  shift 2

  local -a argv=("$@")
  local i token next
  local long_key="--${key}"
  local long_key_with_equal="--${key}="

  for ((i = 0; i < ${#argv[@]}; i++)); do
    token="${argv[i]}"
    if [[ "${token}" == "${long_key}" ]]; then
      if ((i + 1 < ${#argv[@]})); then
        next="${argv[i + 1]}"
        printf '%s' "${next}"
        return 0
      fi
      break
    fi
    if [[ "${token}" == "${long_key_with_equal}"* ]]; then
      printf '%s' "${token#${long_key_with_equal}}"
      return 0
    fi
  done

  printf '%s' "${default_value}"
}

has_flag() {
  local key="$1"
  shift

  local -a argv=("$@")
  local token
  local long_key="--${key}"
  local long_key_with_equal="--${key}="
  local value

  for token in "${argv[@]}"; do
    if [[ "${token}" == "${long_key}" ]]; then
      return 0
    fi
    if [[ "${token}" == "${long_key_with_equal}"* ]]; then
      value="${token#${long_key_with_equal}}"
      case "${value}" in
        0|false|False|FALSE|no|off)
          return 1
          ;;
        *)
          return 0
          ;;
      esac
    fi
  done
  return 1
}

is_train_py_process() {
  local -a argv=("$@")
  local token
  for token in "${argv[@]}"; do
    case "${token}" in
      train.py|*/train.py|train_ablation.py|*/train_ablation.py)
        return 0
        ;;
    esac
  done
  return 1
}

resolve_python_bin() {
  local venv_python="${REPO_ROOT}/.venv/bin/python"
  if [[ -x "${venv_python}" ]]; then
    printf '%s' "${venv_python}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$(command -v python3)"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    printf '%s' "$(command -v python)"
    return 0
  fi
  return 1
}

resolve_config_path() {
  local raw_path="$1"
  local abs_dir

  if [[ -z "${raw_path}" ]]; then
    printf ''
    return 0
  fi

  if [[ "${raw_path}" = /* ]]; then
    printf '%s' "${raw_path}"
    return 0
  fi

  if [[ -f "${raw_path}" ]]; then
    abs_dir="$(CDPATH= cd -- "$(dirname -- "${raw_path}")" && pwd)"
    printf '%s/%s' "${abs_dir}" "$(basename -- "${raw_path}")"
    return 0
  fi

  printf '%s/%s' "${REPO_ROOT}" "${raw_path}"
}

get_parent_pid() {
  local pid="$1"
  awk '/^PPid:/{print $2; exit}' "/proc/${pid}/status" 2>/dev/null
}

collect_launcher_config_paths_from_ancestors() {
  local start_pid="$1"
  local current_pid="${start_pid}"
  local parent_pid depth=0 raw_config_path resolved_config_path
  local -A seen_pids=()
  local -A seen_configs=()
  local -a ancestor_argv=()

  while ((depth < 32)); do
    parent_pid="$(get_parent_pid "${current_pid}" || true)"
    if [[ ! "${parent_pid}" =~ ^[0-9]+$ ]] || ((parent_pid <= 0)); then
      break
    fi
    if [[ -n "${seen_pids["${parent_pid}"]+x}" ]]; then
      break
    fi
    seen_pids["${parent_pid}"]=1

    if [[ -r "/proc/${parent_pid}/cmdline" ]] && mapfile -d '' -t ancestor_argv < "/proc/${parent_pid}/cmdline" 2>/dev/null; then
      if ((${#ancestor_argv[@]} > 0)); then
        raw_config_path="$(extract_flag "config" "" "${ancestor_argv[@]}")"
        if [[ -n "${raw_config_path}" ]]; then
          resolved_config_path="$(resolve_config_path "${raw_config_path}")"
          if [[ -r "${resolved_config_path}" ]] && [[ -z "${seen_configs["${resolved_config_path}"]+x}" ]]; then
            seen_configs["${resolved_config_path}"]=1
            printf '%s\n' "${resolved_config_path}"
          fi
        fi
      fi
    fi

    if ((parent_pid <= 1)); then
      break
    fi
    current_pid="${parent_pid}"
    depth=$((depth + 1))
  done
}

build_schedule_rows_from_config() {
  local config_path="$1"
  local python_bin

  if ! python_bin="$(resolve_python_bin)"; then
    echo "[ERROR] python interpreter is required to parse launcher config." >&2
    return 1
  fi

  "${python_bin}" - "${REPO_ROOT}" "${config_path}" <<'PY'
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
config_arg = Path(sys.argv[2])
if config_arg.is_absolute():
    config_path = config_arg
else:
    repo_candidate = (repo_root / config_arg)
    cwd_candidate = (Path.cwd() / config_arg)
    config_path = repo_candidate if repo_candidate.exists() else cwd_candidate

sys.path.insert(0, str((repo_root / "scripts").resolve()))

try:
    import train_launcher_from_config as launcher
except Exception as exc:
    raise SystemExit(f"[ERROR] Failed to import launcher helper: {exc}") from exc

try:
    config = launcher.load_config(config_path)
    datasets = launcher.validate_config_shape(config)
    mode = config["mode"]
    device = int(config.get("device", 0))

    runs = []
    if mode == "single":
        config["dataset"] = datasets[0]
        runs = launcher.build_single_schedule(config, device)
    else:
        runs = launcher.build_multi_schedule(config, device, datasets)

    for run in runs:
        dataset = str(run["preset"]["dataset"])
        conn_num = int(run["conn_num"])
        label_mode = str(run["label_mode"])
        dist_aux_loss = str(run["dist_aux_loss"])
        experiment_name = launcher.build_experiment_output_name(
            conn_num=conn_num,
            label_mode=label_mode,
            dist_aux_loss=dist_aux_loss,
            conn_layout=run.get("conn_layout"),
            conn_fusion=run.get("conn_fusion", "none"),
            # legacy fusion token removed; do not include it
            fusion_residual_scale=float(run.get("fusion_residual_scale", 0.2)),
            use_seg_aux=bool(run.get("use_seg_aux", False)),
            seg_aux_weight=run.get("seg_aux_weight"),
            experiment_name_suffix=str(run.get("experiment_name_suffix") or ""),
            fusion_branches=str(run.get("fusion_branches", "none")),
        )
        output_dir = str(run.get("output_dir", config.get("output_dir", "output")))
        target_fold = run.get("target_fold")
        if target_fold is None:
            target_fold = ""
        print(f"{dataset}\t{experiment_name}\t{output_dir}\t{target_fold}")
except Exception as exc:
    raise SystemExit(f"[ERROR] Failed to build schedule from config {config_path}: {exc}") from exc
PY
}

build_experiment_candidate_dirs() {
  local output_base="$1"
  local dataset="$2"
  local experiment_name="$3"
  local output_base_name output_parent_name

  output_base_name="$(basename -- "${output_base}")"
  output_parent_name="$(basename -- "$(dirname -- "${output_base}")")"

  if [[ "${output_base_name}" == "${experiment_name}" ]]; then
    printf '%s\n' "${output_base}"
  fi
  if [[ "${output_base_name}" == "${dataset}" ]]; then
    printf '%s\n' "${output_base}/${experiment_name}"
  fi
  if [[ "${output_parent_name}" == "${dataset}" && "${output_base_name}" == "${experiment_name}" ]]; then
    printf '%s\n' "${output_base}"
  fi

  printf '%s\n' \
    "${output_base}/${dataset}/${experiment_name}" \
    "${output_base}/${experiment_name}" \
    "${output_base}"
}

has_final_results_in_dir() {
  local run_dir="$1"
  local fold_glob

  if [[ -r "${run_dir}/final_results.csv" ]]; then
    return 0
  fi

  fold_glob="${run_dir}/final_results_"'*.csv'
  if compgen -G "${fold_glob}" >/dev/null 2>&1; then
    return 0
  fi

  return 1
}

is_experiment_completed() {
  local output_dir_raw="$1"
  local dataset="$2"
  local experiment_name="$3"
  local output_dir_norm output_base
  local candidate_dir
  local -A seen_dirs=()

  output_dir_norm="$(normalize_output_dir "${output_dir_raw}")"
  if [[ "${output_dir_norm}" = /* ]]; then
    output_base="${output_dir_norm}"
  else
    output_base="${REPO_ROOT}/${output_dir_norm}"
  fi

  while IFS= read -r candidate_dir; do
    [[ -n "${candidate_dir}" ]] || continue
    if [[ -n "${seen_dirs["${candidate_dir}"]+x}" ]]; then
      continue
    fi
    seen_dirs["${candidate_dir}"]=1
    if has_final_results_in_dir "${candidate_dir}"; then
      return 0
    fi
  done < <(build_experiment_candidate_dirs "${output_base}" "${dataset}" "${experiment_name}")

  return 1
}

print_schedule_status_for_config() {
  local resolved_config_path="$1"
  local schedule_rows total_experiments running_experiments completed_experiments remaining_experiments
  local schedule_key remaining_row
  local sched_dataset sched_experiment sched_output_dir sched_target_fold
  local -a remaining_rows=()

  if ! schedule_rows="$(build_schedule_rows_from_config "${resolved_config_path}")"; then
    echo "[WARN] Failed to build schedule from inferred config: ${resolved_config_path}" >&2
    return 1
  fi

  total_experiments=0
  running_experiments=0
  completed_experiments=0
  remaining_experiments=0

  while IFS=$'\t' read -r sched_dataset sched_experiment sched_output_dir sched_target_fold; do
    [[ -n "${sched_dataset}" ]] || continue

    total_experiments=$((total_experiments + 1))
    schedule_key="${sched_dataset}|${sched_experiment}|${sched_target_fold}"
    if [[ -n "${running_schedule_keys["${schedule_key}"]+x}" ]]; then
      running_experiments=$((running_experiments + 1))
      continue
    fi

    if is_experiment_completed "${sched_output_dir}" "${sched_dataset}" "${sched_experiment}"; then
      completed_experiments=$((completed_experiments + 1))
      continue
    fi

    remaining_experiments=$((remaining_experiments + 1))
    if [[ -z "${sched_target_fold}" ]]; then
      sched_target_fold="-"
    fi
    printf -v remaining_row '%s\t%s\t%s' "${sched_dataset}" "${sched_experiment}" "${sched_target_fold}"
    remaining_rows+=("${remaining_row}")
  done <<< "${schedule_rows}"

  echo
  echo "Schedule status (${resolved_config_path}):"
  echo "Remaining/Total: ${remaining_experiments}/${total_experiments} (running=${running_experiments}, completed=${completed_experiments})"

  CONFIG_REMAINING_COUNT["${resolved_config_path}"]="${remaining_experiments}"

  if ((remaining_experiments > 0)); then
    {
      printf 'DATASET\tEXPERIMENT\tTARGET_FOLD\n'
      printf '%s\n' "${remaining_rows[@]}"
    } | {
      if command -v column >/dev/null 2>&1; then
        column -t -s $'\t'
      else
        cat
      fi
    }
  else
    echo "No remaining experiments in this config."
  fi

  return 0
}

INCLUDE_ALL=0
SHORT=0
while (($#)); do
  case "$1" in
    --all)
      INCLUDE_ALL=1
      shift
      ;;
    --short)
      SHORT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found in PATH." >&2
  exit 127
fi

if ! GPU_QUERY="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null)"; then
  echo "[ERROR] Failed to query GPU metadata via nvidia-smi." >&2
  exit 1
fi

if ! PROC_QUERY="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "[ERROR] Failed to query GPU compute processes via nvidia-smi." >&2
  exit 1
fi

declare -A GPU_INDEX_BY_UUID=()
while IFS=',' read -r raw_index raw_uuid; do
  index="$(trim "${raw_index}")"
  uuid="$(trim "${raw_uuid}")"
  if [[ -n "${uuid}" ]]; then
    GPU_INDEX_BY_UUID["${uuid}"]="${index}"
  fi
done <<< "${GPU_QUERY}"

declare -a rows=()
declare -a row_keys=()
declare -A seen_pid_gpu=()
declare -A running_schedule_keys=()
declare -A inferred_config_paths=()
declare -A CONFIG_REMAINING_COUNT=()
declare -A ROW_CONFIG_PATH=()
declare -A ROW_ETA_SEC=()
declare -A ROW_PREDICTED_TOTAL_SEC=()
train_process_count=0

while IFS=',' read -r raw_pid raw_gpu_uuid raw_mem; do
  pid="$(trim "${raw_pid}")"
  gpu_uuid="$(trim "${raw_gpu_uuid}")"
  gpu_mem_mb="$(trim "${raw_mem}")"

  [[ -n "${pid}" ]] || continue
  [[ "${pid}" =~ ^[0-9]+$ ]] || continue

  key="${pid}:${gpu_uuid}"
  if [[ -n "${seen_pid_gpu["${key}"]+x}" ]]; then
    continue
  fi
  seen_pid_gpu["${key}"]=1

  cmdline_path="/proc/${pid}/cmdline"
  if [[ ! -r "${cmdline_path}" ]]; then
    continue
  fi

  if ! mapfile -d '' -t argv < "${cmdline_path}" 2>/dev/null; then
    continue
  fi
  ((${#argv[@]} > 0)) || continue

  is_train_process=0
  if is_train_py_process "${argv[@]}"; then
    is_train_process=1
    train_process_count=$((train_process_count + 1))
  elif ((INCLUDE_ALL == 0)); then
    continue
  fi

  dataset="$(extract_flag "dataset" "retouch-Spectrailis" "${argv[@]}")"
  conn_num="$(extract_flag "conn_num" "8" "${argv[@]}")"
  conn_layout="$(extract_flag "conn_layout" "" "${argv[@]}")"
  conn_fusion="$(extract_flag "conn_fusion" "none" "${argv[@]}")"
  legacy_fusion_token=""
  fusion_residual_scale="$(extract_flag "fusion_residual_scale" "0.2" "${argv[@]}")"
  label_mode="$(extract_flag "label_mode" "binary" "${argv[@]}")"
  dist_aux_loss="$(extract_flag "dist_aux_loss" "smooth_l1" "${argv[@]}")"
  use_seg_aux="0"
  if has_flag "use_seg_aux" "${argv[@]}"; then
    use_seg_aux="1"
  fi
  seg_aux_weight="$(extract_flag "seg_aux_weight" "" "${argv[@]}")"
  experiment_name_suffix="$(extract_flag "experiment_name_suffix" "" "${argv[@]}")"
  target_fold="$(extract_flag "target_fold" "" "${argv[@]}")"
  epochs="$(extract_flag "epochs" "45" "${argv[@]}")"
  output_dir_raw="$(extract_flag "output_dir" "output/" "${argv[@]}")"
  device="$(extract_flag "device" "" "${argv[@]}")"
  gpu_index="${GPU_INDEX_BY_UUID["${gpu_uuid}"]:-unknown}"
  output_dir_norm="$(normalize_output_dir "${output_dir_raw}")"
  fusion_branches="$(extract_flag "fusion_branches" "io" "${argv[@]}")"

  if [[ "${output_dir_norm}" = /* ]]; then
    output_base="${output_dir_norm}"
  else
    output_base="${REPO_ROOT}/${output_dir_norm}"
  fi
  experiment_name="$(build_experiment_name "${label_mode}" "${conn_num}" "${dist_aux_loss}" "${conn_layout}" "${conn_fusion}" "${legacy_fusion_token}" "${use_seg_aux}" "${seg_aux_weight}" "${fusion_residual_scale}" "${fusion_branches}")"
  if [[ -n "${experiment_name_suffix}" ]]; then
    experiment_name="${experiment_name}_${experiment_name_suffix}"
  fi
  results_csv="$(resolve_results_csv_path "${output_base}" "${dataset}" "${experiment_name}" "${experiment_name_suffix}")"

  if [[ -r "${results_csv}" ]]; then
    experiment_name_from_path="$(basename -- "$(dirname -- "${results_csv}")")"
    if parsed_experiment_fields="$(parse_experiment_name_fields "${experiment_name_from_path}")"; then
      IFS=$'\t' read -r label_mode conn_num conn_layout dist_aux_loss conn_fusion <<< "${parsed_experiment_fields}"
      experiment_name="${experiment_name_from_path}"
    fi
  fi

  eta_fields="$(compute_eta_fields "${results_csv}" "${epochs}")"
  IFS=$'\t' read -r epoch_progress eta_duration eta_finish eta_status eta_sec_raw predicted_total_sec_raw <<< "${eta_fields}"

  if [[ -z "${device}" ]]; then
    device="${gpu_index}"
  fi

  printf -v row '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "${pid}" \
    "${gpu_index}" \
    "${dataset}" \
    "${conn_num}" \
    "${conn_fusion}" \
    "${experiment_name_suffix:--}" \
    "${label_mode}" \
    "${dist_aux_loss}" \
    "${device}" \
    "${gpu_mem_mb}" \
    "${epoch_progress}" \
    "${eta_duration}" \
    "${eta_finish}" \
    "${eta_status}"
  rows+=("${row}")
  row_keys+=("${key}")
  ROW_ETA_SEC["${key}"]="${eta_sec_raw}"
  ROW_PREDICTED_TOTAL_SEC["${key}"]="${predicted_total_sec_raw}"
  if ((is_train_process == 1)); then
    running_schedule_keys["${dataset}|${experiment_name}|${target_fold}"]=1
    row_config_path=""

    while IFS= read -r inferred_config_path; do
      [[ -n "${inferred_config_path}" ]] || continue
      if [[ -z "${row_config_path}" ]]; then
        row_config_path="${inferred_config_path}"
      fi
      inferred_config_paths["${inferred_config_path}"]=1
    done < <(collect_launcher_config_paths_from_ancestors "${pid}")
    ROW_CONFIG_PATH["${key}"]="${row_config_path}"
  fi
done <<< "${PROC_QUERY}"

if ((SHORT == 0)) && ((train_process_count > 0)); then
  if ((${#inferred_config_paths[@]} == 0)); then
    echo
    echo "[INFO] Could not infer launcher config from active train.py process ancestry."
    echo "       Remaining schedule summary is skipped."
  else
    mapfile -t inferred_config_list < <(printf '%s\n' "${!inferred_config_paths[@]}" | sort)
    for inferred_config_path in "${inferred_config_list[@]}"; do
      if ! print_schedule_status_for_config "${inferred_config_path}"; then
        exit 2
      fi
    done
  fi
fi

if ((${#rows[@]} == 0)); then
  echo "No active train.py GPU process found."
  if ((INCLUDE_ALL == 0)); then
    echo "Tip: run with --all to include non-train GPU compute processes."
  fi
else
  echo
  rendered_rows=()
  for idx in "${!rows[@]}"; do
    row="${rows[idx]}"
    row_key="${row_keys[idx]}"
    IFS=$'\t' read -r pid gpu dataset conn_num conn_fusion exp_suffix label_mode dist_aux_loss device gpu_mem_mb epoch_progress eta_duration eta_finish eta_status <<< "${row}"
    eta_finish_total="n/a"

    row_config_path="${ROW_CONFIG_PATH["${row_key}"]:-}"
    eta_sec_raw="${ROW_ETA_SEC["${row_key}"]:-n/a}"
    predicted_total_sec_raw="${ROW_PREDICTED_TOTAL_SEC["${row_key}"]:-n/a}"
    remaining_count=""
    if [[ -n "${row_config_path}" ]]; then
      remaining_count="${CONFIG_REMAINING_COUNT["${row_config_path}"]:-}"
    fi

    if [[ "${eta_sec_raw}" =~ ^[0-9]+$ ]] && [[ "${predicted_total_sec_raw}" =~ ^[0-9]+$ ]] && [[ "${remaining_count}" =~ ^[0-9]+$ ]]; then
      total_eta_sec=$((eta_sec_raw + (predicted_total_sec_raw * remaining_count)))
      finish_epoch_total=$(( $(date +%s) + total_eta_sec ))
      eta_finish_total="$(format_finish_time_from_epoch "${finish_epoch_total}")"
    fi

    printf -v rendered_row '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
      "${pid}" \
      "${gpu}" \
      "${dataset}" \
      "${conn_num}" \
      "${conn_fusion}" \
      "${exp_suffix}" \
      "${label_mode}" \
      "${dist_aux_loss}" \
      "${device}" \
      "${gpu_mem_mb}" \
      "${epoch_progress}" \
      "${eta_duration}" \
      "${eta_finish}" \
      "${eta_finish_total}" \
      "${eta_status}"
    rendered_rows+=("${rendered_row}")
  done

  {
    printf 'PID\tGPU\tDATASET\tCONN_NUM\tCONN_FUSION\tEXP_SUFFIX\tLABEL_MODE\tDIST_AUX_LOSS\tDEVICE\tGPU_MEM_MB\tEPOCH_PROGRESS\tETA_DURATION\tETA_FINISH\tETA_FINISH_TOTAL\tETA_STATUS\n'
    printf '%s\n' "${rendered_rows[@]}"
  } | {
    if command -v column >/dev/null 2>&1; then
      column -t -s $'\t'
    else
      cat
    fi
  }
fi
