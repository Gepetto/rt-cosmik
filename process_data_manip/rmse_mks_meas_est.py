import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
meas = pd.read_csv('mks_mocap/mks_mocap_test_2_corrected.csv')
est = pd.read_csv('mks_mocap/mks_mocap_model_test_2.csv')

# Initialize RMSE dictionary
rmse_dict = {}

# Number of columns per figure
plots_per_figure = 6
rows, cols = 2, 3  # 2 rows and 3 columns per subplot

# Iterate through each column (assuming both DataFrames have the same structure)
num_cols = meas.shape[1]
fig_count = 0

for i in range(1, num_cols+1):
    print(i)
    # Calculate RMSE for the current column
    rmse = np.sqrt(np.mean((meas.values[:, i-1] - est.values[:, i-1]) ** 2))
    rmse_dict[meas.columns[i-1]] = rmse

    # Plotting
    if (i - 1) % plots_per_figure == 0:
        if i > 1:  # Close previous figure
            plt.tight_layout()
            plt.show()
        fig_count += 1
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()

    ax = axes[(i - 1) % plots_per_figure]
    ax.plot(meas.values[:, i-1], label='mks_meas', linestyle='-')
    ax.plot(est.values[:, i-1], label='mks_est', linestyle='--')
    ax.set_title(f"{meas.columns[i-1]} (RMSE: {rmse:.4f})")
    ax.legend()
    ax.grid()

# Show the last figure
plt.tight_layout()
plt.show()

# Display RMSE values
print("RMSE for each column:")
for key, value in rmse_dict.items():
    print(f"{key}: {value:.4f}")

# Calculate the average RMSE
average_rmse = sum(rmse_dict.values()) / len(rmse_dict)
print(f"\nAverage RMSE: {average_rmse:.4f}")
