import sys
import pandas as pd

video_path = sys.argv[1]

data = pd.read_csv(video_path)

n_lines = len(list(data.iloc[:,0]))

delta_t = pd.Timestamp(data.iloc[-1,1])-pd.Timestamp(data.iloc[1,1])

fps = n_lines / delta_t.total_seconds()

print("FPS :", fps)