import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mocap = pd.read_csv('mks_mocap_downsampled_33Hz.csv')
lstm = pd.read_csv('augmented_mks_interpolated_33hz.csv')
# for i in range(1,y.shape[1]):
#     plt.figure()
#     x=np.arange(0, y.shape[0],1)
#     #plt.title(f"{'_'.join(y.columns[i])}")
#     #plt.plot(y.values[:, i], 'black', label='ref')
#     plt.plot(y.values[212:, i], label='mks_mocap', linestyle='-')
#     plt.plot(lstm.values[:, i], label='mks_lstm', linestyle='--')
#     plt.legend()
#     plt.grid()
#     plt.show()


columns_to_plot_mocap = ["r_mwrist_study_x", "r_mwrist_study_y", "r_mwrist_study_z"]  # List of columns to plot
columns_to_plot_lstm = ["L_mwrist_study_x", "L_mwrist_study_y", "L_mwrist_study_z"]
# for col in columns_to_plot:
#     if col in mocap.columns and col in lstm.columns:
#         plt.figure(figsize=(10, 6))
#         plt.plot(mocap[col].values[212:], label='mocap', color='blue')
#         plt.plot(lstm[col].values, label='lstm', color='orange', linestyle='--')
#         plt.ylabel(col)
#         plt.title(f"Comparison of {col} between mks_mocap and mks_lstm")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#     else:
#         print(f"Column '{col}' not found in one or both files.")


for col_mocap, col_lstm in zip(columns_to_plot_mocap, columns_to_plot_lstm):
    if col_mocap in mocap.columns and col_lstm in lstm.columns:
        plt.figure(figsize=(10, 6))
        plt.plot( mocap[col_mocap].values[212:], label=f'Mocap - {col_mocap}', color='blue')
        plt.plot(lstm[col_lstm].values, label=f'LSTM - {col_lstm}', color='orange', linestyle='--')
        plt.ylabel('mks')
        plt.title(f"Comparison of {col_mocap} (Mocap) and {col_lstm} (LSTM)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Column '{col_mocap}' not found in mocap or '{col_lstm}' not found in lstm.")


# common_columns = set(mocap.columns).intersection(lstm.columns)
# common_columns.remove('time')  # Exclude 'time' column if present
# print(len(common_columns))
# input()

# for col in sorted(common_columns):  # Sorting for consistent order
#     print(col)
#     plt.figure(figsize=(10, 6))  # Create a new figure for each column
#     plt.plot(mocap[col].values[212:], label=f'mocap')
#     plt.plot(lstm[col].values, label=f'lstm', linestyle='--')

#     # Add labels, title, and legend for each plot
#     plt.xlabel("Time")
#     plt.ylabel("Value")
#     plt.title(f"{col}")
#     plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), ncol=1)
#     plt.grid(True)

#     # Adjust layout and show the plot for each column
#     plt.show()