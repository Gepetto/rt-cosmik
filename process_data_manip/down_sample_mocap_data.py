import pandas as pd
import time
# Load the CSV file
file_path = 'q/q_mocap_qp.csv'  # Replace with your file path
output_file_path = 'q/q_mocap_qp_downsampled_33Hz.csv'  # Replace with your desired output file path

data = pd.read_csv(file_path)

# Downsample the data to 30 Hz by selecting every third row

downsampled_data = data.iloc[::3]
print(downsampled_data)
# Save the downsampled data to a new CSV file


df = pd.DataFrame(downsampled_data)
df.to_csv(output_file_path, index=True)


