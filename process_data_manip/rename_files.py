import os

nbr_cam = 2
no_trial = "Nicolas"
task_list = [
             "robot_sanding","robot_welding",
             "sanding","sanding_sat","sit_to_stand","squat","upper","walk","walk_front","welding","welding_sat"]

base_path = "/root/workspace/ros_ws/src/rt-cosmik/output/"

for task in task_list:
    old_name = os.path.join(base_path, f"{no_trial}/cosmik_2cams/{task}/q_cosmik_ipopt_2_finetuned.csv")
    new_name = os.path.join(base_path, f"{no_trial}/cosmik_2cams/{task}/q_cosmik_ipopt_2.csv")

# rename
    os.rename(old_name, new_name)
    print(f"Renamed '{old_name}' to '{new_name}'")
