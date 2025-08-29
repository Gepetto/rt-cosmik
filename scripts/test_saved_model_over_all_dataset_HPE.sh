#!/bin/bash
set -euo pipefail

read -p "Use mocap (T/F)? " use_mocap
read -p "Add noise (T/F)? " add_noise
read -p "Use weights (T/F)? " use_weights
read -p "Rotation probability (0-1)? " rot_prob
read -p "Rotation max degree (0-180)? " rot_max_deg
read -p "Rotation scheme (off/prob/det)? " rotation_scheme
read -p "Rotation number ? " rotation_number

# --- config ---
dataset_path="/home/ngouget/Codes/datasets/COSMIK_dataset"
MAX_JOBS=6
RESULTS_FILE="../../Results/Tests_perfos_LSTM/detailed_rmse/resultsHPE_ftF_alT_mT_n${add_noise}_w${use_weights}_prot${rot_prob}_maxrot${rot_max_deg}_rotscheme${rotation_scheme}_nrot${rotation_number}.csv"
LOG_DIR="../../Results/Tests_perfos_LSTM/logsHPE_ftF_alT_mT_n${add_noise}_w${use_weights}_prot${rot_prob}_maxrot${rot_max_deg}_rotscheme${rotation_scheme}_nrot${rotation_number}"
mkdir -p "$LOG_DIR"

# header du CSV
echo "subject,trial,average_rmse,average_rmse_OpenCap" > "$RESULTS_FILE"

wait_for_slot() {
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 0.2
  done
}

have_flock=1
command -v flock >/dev/null 2>&1 || have_flock=0
mutex_fd=200

for subject_dir in "$dataset_path"/*; do
  [ -d "$subject_dir" ] || continue
  subject="$(basename "$subject_dir")"

  if [ $subject != "Flavie" ] && [ $subject = "Kahina" ]; then
        continue
  fi

  for trial_dir in "$subject_dir"/*; do
    [ -d "$trial_dir" ] || continue
    trial="$(basename "$trial_dir")"

    wait_for_slot

    (
      set -euo pipefail

      # 1) run augmenter
      if ! python run_marker_augmenter.py --subject $subject --trial $trial --use-mocap $use_mocap --add-noise $add_noise --fine-tune F --add-layer T \
 --use-weights $use_weights --rot-prob $rot_prob --rot-max-deg $rot_max_deg --rotation-scheme $rotation_scheme --n-rotations $rotation_number; then
        echo "[FAIL] run_marker_augmenter.py $subject $trial" >&2
        python run_marker_augmenter.py --subject $subject --trial $trial --use-mocap $use_mocap --add-noise $add_noise --fine-tune F --add-layer T \
 --use-weights $use_weights --rot-prob $rot_prob --rot-max-deg $rot_max_deg --rotation-scheme $rotation_scheme --n-rotations $rotation_number \ 
 >"$LOG_DIR/${subject}__${trial}_augmenter.log" 2>&1 || true
        exit 1
      fi

      # 2) run checker
      results_line="$(python ../process_data_manip/z_check_multiple_mks_gp_v.py --subject $subject --trial $trial --use-mocap $use_mocap --add-noise $add_noise --fine-tune F --add-layer T \
 --use-weights $use_weights --rot-prob $rot_prob --rot-max-deg $rot_max_deg --rotation-scheme $rotation_scheme --n-rotations $rotation_number \
 2> >(tee "$LOG_DIR/${subject}__${trial}_checker.log" >&2) \
                      | grep -E '^RESULTS[, ]' || true)"

      if [ -z "$results_line" ]; then
        echo "[FAIL] No RESULTS line for $subject/$trial (see $LOG_DIR/${subject}__${trial}_checker.log)" >&2
        exit 1
      fi

      # transformer espaces en virgules
      results_line="${results_line// /,}"
      IFS=',' read -r _ avg_rmse avg_rmse_opencap <<< "$results_line"

      # append sécurisé
      if [ "$have_flock" -eq 1 ]; then
        exec {mutex_fd}>>"$RESULTS_FILE"
        flock "$mutex_fd"
        echo "$subject,$trial,$avg_rmse,$avg_rmse_opencap" >> "$RESULTS_FILE"
        flock -u "$mutex_fd"
        exec {mutex_fd}>&-
      else
        echo "$subject,$trial,$avg_rmse,$avg_rmse_opencap" >> "$RESULTS_FILE"
      fi

    ) &
  done
done

wait
echo "✓ All subjects/trials processed. Results in $RESULTS_FILE"
