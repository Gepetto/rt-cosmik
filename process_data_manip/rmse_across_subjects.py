import os
import pandas as pd
from glob import glob

data_path = "/root/workspace/ros_ws/src/rt-cosmik/output"  


#excel per subject
excel_files = []
for subject in os.listdir(data_path):
    results_dir = os.path.join(data_path, subject, "results")
    excel_file = os.path.join(results_dir, f"rmse_par_dof_over_trials_{subject}.xlsx")
    if os.path.exists(excel_file):
        excel_files.append(excel_file)

dfs = []
all_trials = set()
for f in excel_files:
    df = pd.read_excel(f, sheet_name="per_trial", header=[0,1], index_col=0)
    dfs.append(df)
    # get trials name
    all_trials.update([col[0] for col in df.columns])

skip_trial = "overhead_front"  # the trial you want to skip

dfs = []
all_trials = []
for f in excel_files:
    df = pd.read_excel(f, sheet_name="per_trial", header=[0,1], index_col=0)
    dfs.append(df)
    # garder l'ordre d'apparition des trials en sautant le skip_trial
    for col in df.columns.get_level_values(0).unique():
        if col not in all_trials and col != skip_trial:
            all_trials.append(col)
            
metrics = ['rmse_deg', 'corr']
full_columns = pd.MultiIndex.from_product([all_trials, metrics])

dfs_aligned = []
for df in dfs:
    df_aligned = df.reindex(columns=full_columns)
    dfs_aligned.append(df_aligned)

all_data = pd.concat(dfs_aligned, keys=[os.path.basename(f).split("_")[4] for f in excel_files], names=["subject"])

#mea per dof across all subject
mean_across_subjects = all_data.groupby(level=1).mean()

dofs_to_use = ['Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
               'Lumbar_flex_ext', 'Lumbar_lateral_flex','Lcalvicule_x',
               'Lshoulder_flex_ext','Lshoulder_abd_add','Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi',
               'Cervical_flex_ext','Cervical_lat_bend','Cervical_int_ext_rot','rcalvicule_x',
               'Rshoulder_flex_ext','Rshoulder_abd_add','Rshoulder_int_ext_rot','Relbow_flex_ext','Relbow_pron_supi',
               'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot','Rknee_flex_ext','Rankle_flex_ext','Rankle_abd_add']

mean_across_subjects = mean_across_subjects.reindex(dofs_to_use + ["mean", "std"])

std_across_subjects = all_data.groupby(level=1).std().reindex(dofs_to_use + ["mean", "std"])

output_file = os.path.join(data_path, "ipopt_rmse_per_dof_over_trials_all_subjects.xlsx")
with pd.ExcelWriter(output_file) as writer:
    mean_across_subjects.to_excel(writer, sheet_name="mean_across_subjects")

print(f"Fichier moyen généré : {output_file}")
