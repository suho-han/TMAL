#!/usr/bin/env bash
set -euo pipefail

# remove_and_rename_fusion_profiles.sh
# Dry-run by default. Use --apply to perform deletions/renames.

APPLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=true; shift ;;
    -h|--help) echo "Usage: $0 [--apply]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

ROOT_DIR="$(pwd)"
OUTPUT_DIRS=(output output_mnt output_smoke)

echo "Running in: $ROOT_DIR"
echo "Dry-run mode: ${APPLY:-false}" 

list_bc_dirs() {
  for droot in "${OUTPUT_DIRS[@]}"; do
    if [[ -d "$droot" ]]; then
      find "$droot" -maxdepth 3 -type d -regextype posix-extended \
        -regex '.*/[^/]*_([BC])_[^/]*' -print || true
    fi
  done
}

list_a_dirs() {
  for droot in "${OUTPUT_DIRS[@]}"; do
    if [[ -d "$droot" ]]; then
      find "$droot" -maxdepth 3 -type d -regextype posix-extended \
        -regex '.*/[^/]*_A_[^/]*' -print || true
    fi
  done
}

echo "\nCandidate B/C-profile directories (would be DELETED):"
list_bc_dirs || true

echo "\nCandidate A-profile directories (would be RENAMED by removing '_A_') :"
list_a_dirs || true

if [ "$APPLY" = true ]; then
  echo "\nApplying changes..."

  # Delete B/C dirs
  while IFS= read -r dir; do
    echo "Deleting: $dir"
    rm -rf "$dir"
  done < <(list_bc_dirs)

  # Rename A dirs
  while IFS= read -r dir; do
    base="$dir"
    newbase="$(echo "$base" | sed 's/_A_/_/')"
    if [[ "$base" != "$newbase" ]]; then
      if [[ -e "$newbase" ]]; then
        echo "WARNING: target exists, skipping rename: $base -> $newbase"
      else
        echo "Renaming: $base -> $newbase"
        mv "$base" "$newbase"
      fi
    fi
  done < <(list_a_dirs)

  echo "\nDone."
else
  echo "\nDry-run complete. Rerun with --apply to perform deletions/renames." 
fi

exit 0
