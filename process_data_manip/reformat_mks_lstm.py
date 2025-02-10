import pandas as pd
import numpy as np



input_file_path = 'mks_lstm/augmented_markers_positions_test_2.csv'  
output_file_path = 'augmented_markers_positions_by_rows.csv'
# header = [
#     'Time', 'r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study', 'L.PSIS_study', 'r_knee_study',
#     'r_mknee_study', 'r_ankle_study', 'r_mankle_study', 'r_toe_study', 'r_5meta_study',
#     'r_calc_study', 'L_knee_study', 'L_mknee_study', 'L_ankle_study', 'L_mankle_study',
#     'L_toe_study', 'L_calc_study', 'L_5meta_study', 'r_shoulder_study', 'L_shoulder_study',
#     'C7_study', 'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study', 'L_thigh1_study',
#     'L_thigh2_study', 'L_thigh3_study', 'r_sh1_study', 'r_sh2_study', 'r_sh3_study',
#     'L_sh1_study', 'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study', 'r_lelbow_study',
#     'r_melbow_study', 'r_lwrist_study', 'r_mwrist_study', 'L_lelbow_study', 'L_melbow_study',
#     'L_lwrist_study', 'L_mwrist_study'
# ]

# Open and process the file
with open(input_file_path, 'r') as file:
    # Read all lines and skip the header
    lines = file.readlines()[1:]
    
# Group every 42 lines and extract numerical data
grouped_data = []
for i in range(0, len(lines), 43):
    group = lines[i:i + 43]
    # Get the 'Time' column from the first row of the group
    time = group[0].split(",")[1].strip()
    # Extract numerical values (columns after the 4th comma)
    numeric_data = np.array([
        np.array(row.split(",")[3:], dtype=float) for row in group
    ]).flatten()
    # Combine the time and numerical data into a single row
    grouped_data.append([time] + numeric_data.tolist())

# Create a DataFrame for structured output
dynamic_header = ['Time'] + [f"Marker_{i}" for i in range(1,  len(grouped_data[0]))]

df = pd.DataFrame(grouped_data, columns=dynamic_header)

# Save the processed data to a new file
df.to_csv(output_file_path, index=False)

print(f"Processed data with time saved to {output_file_path}")
