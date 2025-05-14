#get barycenter using mks data to fit with pose qr code.
import numpy as np
import pandas as pd

from utils import *

no_test = "calib_mocap_2_cam1"
# df = pd.read_csv(f"output/test{no_test}/mks_data.csv")

df = pd.read_csv(f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_test}/mks_data.csv")
mks_array = np.array([list(map(float, row.split(";"))) for row in df["mks_data"]])
barycenter_global_list = []
  
for i in range (len(mks_array)):
    A = mks_array[i, :3]
    B = mks_array[i, 3:6]
    C = mks_array[i, 6:9]
    R = calculate_frame(A, B, C)
    print(A)
    print(C)

    barycenter = (A + C) / 2
    # print(barycenter)
    barycenter_local_frame = transform_to_local_frame(barycenter, B, R)
    barycenter_local_frame[2] = barycenter_local_frame[2]- 0.01
    # print(barycenter_local_frame)


    barycenter_global_frame = transform_to_global_frame(barycenter_local_frame, B, R)
    # print(barycenter_global_frame)
    barycenter_global_list.append(barycenter_global_frame)

# Convert to DataFrame and save to CSV
barycenter_global_df = pd.DataFrame(barycenter_global_list, columns=["Bx", "By", "Bz"])
barycenter_global_df.to_csv(f"output/{no_test}/barycenter_raw.csv", index=False)

