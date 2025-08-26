import pandas as pd
import sys
import os
import numpy as np

csv_path = sys.argv[1]

df = pd.read_csv(csv_path)
mean_finetuned = df['average_rmse'].mean()
mean_opencap = df['average_rmse_OpenCap'].mean()

print(f"mean RMSE finetuned: {mean_finetuned}")
print(f"mean RMSE opencap: {mean_opencap}")