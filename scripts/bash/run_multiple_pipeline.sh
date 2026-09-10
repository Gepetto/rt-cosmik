DATASET_DIR="/root/workspace/RT-COSMIK-paper-offline/data"
SEARCH_DIR="$DATASET_DIR/videos"
camera_sets=("0" "0 2" "0 2 4 6")

# Scan all subject folders inside data_new/videos/
for subject_path in "$SEARCH_DIR"/*/; do
  [ -d "$subject_path" ] || continue
  p=$(basename "$subject_path")

  # Scan all task folders inside each subject folder
  for task_path in "$subject_path"*/; do
    [ -d "$task_path" ] || continue
    t=$(basename "$task_path")

    for cams in "${camera_sets[@]}"; do
      python3 ../python/core/run_pipeline_threaded.py --visualizer meshcat\
          --dataset "$DATASET_DIR" --participant "$p" --task "$t" --camera $cams \
          || echo "FAILED $p/$t (camera: $cams)"
    done
  done
done