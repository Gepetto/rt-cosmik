import pandas as pd

# Load the CSV files
csv1 = pd.read_csv('mks_lstm/augmented_markers_positions_by_rows_with_header.csv')
csv2 = pd.read_csv('q_cosmik_qp_modele_mocap.csv')

# Retrieve the first column from the first CSV
first_column = csv1.iloc[:, 0]  # Select the first column by position

# Insert the first column into the second CSV as the first column
csv2.insert(0, first_column.name, first_column)

# Save the updated second CSV to a new file
csv2.to_csv('updated_second_csv_file.csv', index=False)

print("First column from the first CSV has been added to the second CSV!")
