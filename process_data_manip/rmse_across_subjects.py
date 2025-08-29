import os
import pandas as pd
from glob import glob

data_path = "/root/workspace/ros_ws/src/rt-cosmik/output"  


#excel per subject
excel_files = []
for subject in os.listdir(data_path):
    results_dir = os.path.join(data_path, subject, "results")
    excel_file = os.path.join(results_dir, f"swika_mocap_lag_rmse_par_dof_over_trials_{subject}.xlsx")
    if os.path.exists(excel_file):
        excel_files.append(excel_file)

dfs = []
all_trials = []

# liste des trials que tu veux garder
keep_trials = ["bolting", "bolting_sat", "crouch_object","robot_sanding","robot_welding","lifting","hitting","hitting_sat","overhead","sanding"]  # remplace par les noms réels

for f in excel_files:
    df = pd.read_excel(f, sheet_name="per_trial", header=[0,1], index_col=0)
    
    # filtrer les colonnes pour ne garder que les trials désirés
    cols_to_keep = [col for col in df.columns.get_level_values(0) if col in keep_trials]
    df = df.loc[:, df.columns.get_level_values(0).isin(keep_trials)]
    
    dfs.append(df)
    
    # garder l'ordre d'apparition
    for col in cols_to_keep:
        if col not in all_trials:
            all_trials.append(col)

# print("Trials gardées :", all_trials)
            
metrics = ['rmse_deg', 'corr']
full_columns = pd.MultiIndex.from_product([all_trials, metrics])

dfs_aligned = []
for df in dfs:
    df_aligned = df.reindex(columns=full_columns)
    dfs_aligned.append(df_aligned)

all_data = pd.concat(dfs_aligned, keys=[os.path.basename(f).split("_")[4] for f in excel_files], names=["subject"])

#mea per dof across all subject
# mean_across_subjects = all_data.groupby(level=1).mean()

dofs_to_use = ['Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
               'Lumbar_flex_ext', 'Lumbar_lateral_flex','Lcalvicule_x',
               'Lshoulder_flex_ext','Lshoulder_abd_add','Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi',
               'Cervical_flex_ext','Cervical_lat_bend','Cervical_int_ext_rot','rcalvicule_x',
               'Rshoulder_flex_ext','Rshoulder_abd_add','Rshoulder_int_ext_rot','Relbow_flex_ext','Relbow_pron_supi',
               'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot','Rknee_flex_ext','Rankle_flex_ext','Rankle_abd_add']

# mean_across_subjects = mean_across_subjects.reindex(dofs_to_use + ["mean", "std"])

# std_across_subjects = all_data.groupby(level=1).std().reindex(dofs_to_use + ["mean", "std"])
# # Calculer la moyenne des métriques pour chaque DoF
# mean_across_subjects['rmse_mean'] = mean_across_subjects.xs('rmse_deg', level=1, axis=1).mean(axis=1)
# mean_across_subjects['corr_mean'] = mean_across_subjects.xs('corr', level=1, axis=1).mean(axis=1)

# output_file = os.path.join(data_path, "swika_down_rmse_per_dof_over_trials_all_subjects.xlsx")
# with pd.ExcelWriter(output_file) as writer:
#     mean_across_subjects.to_excel(writer, sheet_name="mean_across_subjects")

# print(f"Fichier moyen généré : {output_file}")
# Compute mean and std across subjects for each DOF and each trial
mean_across_subjects = all_data.groupby(level=1).mean()
std_across_subjects = all_data.groupby(level=1).std()

# Reindex to keep only the DOFs you want
mean_across_subjects = mean_across_subjects.reindex(dofs_to_use)
std_across_subjects = std_across_subjects.reindex(dofs_to_use)

# Create new MultiIndex columns to interleave std next to mean for each trial
new_cols = []
for trial in all_trials:
    new_cols.append((trial, 'rmse_deg'))
    new_cols.append((trial, 'rmse_std'))
    new_cols.append((trial, 'corr'))
    new_cols.append((trial, 'corr_std'))

# Build a new DataFrame with interleaved columns
df_combined = pd.DataFrame(index=mean_across_subjects.index, columns=pd.MultiIndex.from_tuples(new_cols))

for trial in all_trials:
    df_combined[(trial, 'rmse_deg')] = mean_across_subjects[(trial, 'rmse_deg')]
    df_combined[(trial, 'rmse_std')] = std_across_subjects[(trial, 'rmse_deg')]
    df_combined[(trial, 'corr')] = mean_across_subjects[(trial, 'corr')]
    df_combined[(trial, 'corr_std')] = std_across_subjects[(trial, 'corr')]

# Save to Excel
output_file = os.path.join(data_path, "swika_down_rmse_per_dof_over_trials_all_subjects.xlsx")
with pd.ExcelWriter(output_file) as writer:
    df_combined.to_excel(writer, sheet_name="mean_std_per_trial")

print(f"Fichier moyen et std par trial généré : {output_file}")
