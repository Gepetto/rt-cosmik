import sys
import pandas as pd

csv_path = sys.argv[1]

data = pd.read_csv(csv_path)

n_lines = len(list(data.iloc[:,0]))

print("Total number of mks data :", n_lines)