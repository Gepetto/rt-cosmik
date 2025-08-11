import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

pretrained_path = sys.argv[1]
body_part = sys.argv[2]

# Load data
data_path = os.path.join(pretrained_path, f"v0.3_{body_part}")
df = pd.read_csv(os.path.join(data_path, "predictions_epoch_295.csv"))
print(df)
list_inds = [ind for ind in range(0,720,64*720)]
plt.plot([i for i in range(1,65)], df.iloc[list_inds*,0], 'ro', label='Truth x')
plt.plot([i for i in range(1,65)], df['prediction'][list_inds*,1], 'bo', label='Predictions x')
plt.savefig(os.path.join(pretrained_path, "pred_vs_gt.png"))