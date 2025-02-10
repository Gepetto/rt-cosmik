import pandas as pd

mks_lstm = 'mks_lstm/augmented_markers_positions_by_rows_with_header.csv'
mocap = pd.read_csv(mks_lstm).iloc[:5, 1:]


# mks_mocap = 'mks_mocap/mks_mocap_test_2.csv'
# mocap = pd.read_csv(mks_mocap).iloc[:5, 1:]


column_means = {}
for col in mocap.columns:
    total = mocap[col].sum()  # Sum of the column values
    count = len(mocap[col])   # Number of rows in the column
    column_means[col] = total / count

# Convert the means dictionary to a DataFrame with one row
means_df = pd.DataFrame([column_means])

# Save the means to a CSV file
# output_file = 'lstm_mean.csv'
# means_df.to_csv(output_file, index=False)

# print(f"Means saved to {output_file}")
# m = pd.read_csv("lstm_mean.csv")
# marker_list = [
#     "r.ASIS_study_x"
# ]
# m = pd.read_csv("lstm_mean.csv", usecols=marker_list)