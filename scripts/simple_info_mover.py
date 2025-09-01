import sys
import os
import shutil

src_dataset_path = sys.argv[1]
dst_dataset_path = sys.argv[2]

for subject in os.listdir(src_dataset_path):
    subject_path = os.path.join(src_dataset_path, subject)
    for trial in os.listdir(subject_path):
        if trial == "info.txt":
            shutil.copy2(os.path.join(src_dataset_path, subject, trial), os.path.join(dst_dataset_path, subject, trial))
