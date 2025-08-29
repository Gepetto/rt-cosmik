import os
import sys
import shutil

src_path = sys.argv[1]
dst_path = sys.argv[2]

cosmik_repo_list = os.listdir(src_path)

for cosmik_repo in cosmik_repo_list:
    subject = cosmik_repo #[7:]
    subject = subject.capitalize()
    print(subject)
    src_subject_path = os.path.join(src_path, cosmik_repo)
    for trial in os.listdir(src_subject_path):
        src_trial_path = os.path.join(src_subject_path, trial)
        dst_trial_path = os.path.join(dst_path, subject, trial)
        for file in os.listdir(src_trial_path):
            file_path = os.path.join(src_trial_path, file)
            shutil.move(file_path, dst_trial_path)
