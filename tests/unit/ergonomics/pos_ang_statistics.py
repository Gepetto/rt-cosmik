import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

angles_csv_path = "/home/ngouget/Codes/rt-cosmik/output/Test_REBA/q_cosmik_ipopt.csv"
positions_csv_path = "/home/ngouget/Codes/rt-cosmik/output/Test_REBA/mks_model_cosmik_ipopt.csv"

data_angles = pd.read_csv(angles_csv_path)
data_positions = pd.read_csv(positions_csv_path)

def transformer(x):
    return np.rad2deg(x)

data_angles = data_angles.applymap(transformer)

# data_angles.plot(y="thoracic_rot_int_ext", marker="o")
# plt.show()

diffr_x = pd.Series(data_positions["r_shoulder_study_x"] - data_positions["r.ASIS_study_x"])
diffr_y = pd.Series(data_positions["r_shoulder_study_y"] - data_positions["r.ASIS_study_y"])
diffr_z = pd.Series(data_positions["r_shoulder_study_z"] - data_positions["r.ASIS_study_z"])

diffl_x = pd.Series(data_positions["L_shoulder_study_x"] - data_positions["L.ASIS_study_x"])
diffl_y = pd.Series(data_positions["L_shoulder_study_y"] - data_positions["L.ASIS_study_y"])
diffl_z = pd.Series(data_positions["L_shoulder_study_z"] - data_positions["L.ASIS_study_z"])

def square(x):
    return x**2

diffr_x_square = diffr_x.apply(square)
diffr_y_square = diffr_y.apply(square)
diffr_z_square = diffr_z.apply(square)

diffl_x_square = diffl_x.apply(square)
diffl_y_square = diffl_y.apply(square)
diffl_z_square = diffl_z.apply(square)

def sqrt(x):
    return np.sqrt(x)

distr_square = pd.Series(diffr_x_square + diffr_y_square + diffr_z_square)
distr = pd.Series(np.sqrt(distr_square))

distl_square = pd.Series(diffl_x_square + diffl_y_square + diffl_z_square)
distl = pd.Series(np.sqrt(distl_square))

print(distr.mean())  
print(distr.std())

print(distl.mean())  
print(distl.std())

dist = pd.DataFrame([distr, distl]).T

print(dist)

dist.plot(marker="o", legend=["right", "left"])
plt.show()