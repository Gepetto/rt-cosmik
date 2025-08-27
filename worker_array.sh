#!/bin/bash -l
set -euo pipefail

# Args
PAIRS_FILE="${1:?pairs.txt missing}"
RESULTS_TMP_DIR="${2:?results_tmp dir missing}"
LOG_DIR="${3:-logs}"
mkdir -p "$RESULTS_TMP_DIR" "$LOG_DIR"

# Expect SLURM_ARRAY_TASK_ID (0-based)
IDX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

# Read subject,trial for this index
IFS=',' read -r subject trial < <(sed -n "$((IDX+1))p" "$PAIRS_FILE")

# Env options (passed from sbatch --export or exported before sbatch)
use_mocap="${use_mocap:?}"
add_noise="${add_noise:?}"
use_weights="${use_weights:?}"
rot_prob="${rot_prob:?}"
rot_max_deg="${rot_max_deg:?}"
rotation_scheme="${rotation_scheme:?}"
rotation_number="${rotation_number:?}"

# --- nettoyage des noms (comme déjà fait) ---
subject="${subject%%[[:space:]]}"; trial="${trial%%[[:space:]]}"
subject="${subject%$'\r'}";        trial="${trial%$'\r'}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate lstm

# 1) run augmenter (silent; log only if fail)
if ! ~/miniforge3/envs/lstm/bin/python scripts/run_marker_augmenter.py \
  --subject "$subject" --trial "$trial" \
  --use-mocap "$use_mocap" --add-noise "$add_noise" \
  --fine-tune F --add-layer T --use-weights "$use_weights" \
  --rot-prob "$rot_prob" --rot-max-deg "$rot_max_deg" \
  --rotation-scheme "$rotation_scheme" --n-rotations "$rotation_number" >/dev/null 2>&1; then
  ~/miniforge3/envs/lstm/bin/python scripts/run_marker_augmenter.py \
    --subject "$subject" --trial "$trial" \
    --use-mocap "$use_mocap" --add-noise "$add_noise" \
    --fine-tune F --add-layer T --use-weights "$use_weights" \
    --rot-prob "$rot_prob" --rot-max-deg "$rot_max_deg" \
    --rotation-scheme "$rotation_scheme" --n-rotations "$rotation_number" \
    >"$LOG_DIR/${subject}__${trial}_augmenter.log" 2>&1 || true
  echo "[FAIL] augmenter $subject/$trial" >&2
  exit 1
fi

# 2) run checker and capture RESULTS line
results_line="$(~/miniforge3/envs/lstm/bin/python process_data_manip/z_check_multiple_mks_gp_v.py \
  --subject "$subject" --trial "$trial" \
  --use-mocap "$use_mocap" --add-noise "$add_noise" \
  --fine-tune F --add-layer T --use-weights "$use_weights" \
  --rot-prob "$rot_prob" --rot-max-deg "$rot_max_deg" \
  --rotation-scheme "$rotation_scheme" --n-rotations "$rotation_number" \
  2> >(tee "$LOG_DIR/${subject}__${trial}_checker.log" >&2) \
  | grep -E '^RESULTS[, ]' || true)"

if [ -z "$results_line" ]; then
  echo "[FAIL] No RESULTS for $subject/$trial (see $LOG_DIR/${subject}__${trial}_checker.log)" >&2
  exit 1
fi

# Normalize spaces → commas; parse
results_line="${results_line// /,}"
IFS=',' read -r _ avg_rmse avg_rmse_opencap <<< "$results_line"

# Write single-row CSV for this pair
echo "subject,trial,average_rmse,average_rmse_OpenCap" > "$RESULTS_TMP_DIR/${subject}__${trial}.csv"
echo "$subject,$trial,$avg_rmse,$avg_rmse_opencap" >> "$RESULTS_TMP_DIR/${subject}__${trial}.csv"

