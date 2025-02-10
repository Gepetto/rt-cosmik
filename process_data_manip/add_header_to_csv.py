# Write the header to a CSV file
import csv

points = [
    "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study", "C7_study",
    "r_knee_study", "r_mknee_study", "L_knee_study", "L_mknee_study",
    "r_ankle_study", "r_mankle_study", "L_ankle_study", "L_mankle_study",
    "r_toe_study", "r_5meta_study", "L_toe_study", "L_5meta_study",
    "r_calc_study", "L_calc_study", "r_shoulder_study", "L_shoulder_study",
    "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
    "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
    "r_sh1_study", "r_sh2_study", "r_sh3_study",
    "L_sh1_study", "L_sh2_study", "L_sh3_study",
    "r_lelbow_study", "r_melbow_study", "L_lelbow_study", "L_melbow_study",
    "r_mwrist_study", "r_lwrist_study", "L_mwrist_study", "L_lwrist_study"
]

points_lstm = [
    "r.ASIS_study", "L.ASIS_study", "r.PSIS_study", "L.PSIS_study",
    "r_knee_study", "r_mknee_study", 
    "r_ankle_study", "r_mankle_study", "r_toe_study", "L_5meta_study",
    "r_calc_study", 
    "L_knee_study", "L_mknee_study",
    "L_ankle_study", "L_mankle_study", "L_toe_study", "L_calc_study","r_5meta_study",
    "r_shoulder_study", "L_shoulder_study",
    "C7_study",
    "r_thigh1_study", "r_thigh2_study", "r_thigh3_study",
    "L_thigh1_study", "L_thigh2_study", "L_thigh3_study",
    "r_sh1_study", "r_sh2_study", "r_sh3_study",
    "L_sh1_study", "L_sh2_study", "L_sh3_study",
    "RHJC_study", "LHJC_study", 

    "r_lelbow_study", "r_melbow_study", "r_lwrist_study", "r_mwrist_study",
    "L_lelbow_study", "L_melbow_study", "L_lwrist_study", "L_mwrist_study"
     
]


# Generate the header
header = ["time"] + [f"{points_lstm}_{axis}" for points_lstm in points_lstm for axis in ["x", "y", "z"]]



input_csv = "augmented_markers_positions_by_rows.csv"  # Replace with your input CSV filename
output_csv = "augmented_markers_positions_by_rows_with_header.csv"    # Replace with your output CSV filename

# Read the input CSV, add the header, and save to a new file
with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    # Skip the first line
    next(reader, None)
    # Write the header
    writer.writerow(header)
    
    # Write the existing rows
    writer.writerows(reader)

print(f"Header added and saved to {output_csv}")
