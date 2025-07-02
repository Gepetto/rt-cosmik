import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
import csv
import eigenpy
import hppfcl
import pinocchio as pin
import numpy as np
import sys 
from tools.robot import Robot
from pinocchio.visualize import GepettoVisualizer
from IK import *
from calibration import *
import pandas as pd
from tools.robotvisualization import display_estimated_markers, display_markers, apply_plug_in_gait_colors,color_segment

def deg2rad(angle):
    return np.pi*angle/180

# TIME SYNC FORCE DATA

# with open('/home/msabbah/Bureau/STAGE/figaro/datasets/Identif/Data_JoB/15_06_2023/LeoH/Mocap/Resampled/Trial0_Resampled.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     no_lines= len(list(spamreader))-1

# forces_trajectories = np.zeros((no_lines,6))

# c=0

# length_fp=1.8
# width_fp=0.9

# FP = pin.SE3(np.eye(3),np.array([length_fp/2,width_fp/2,0]))

# with open('/home/msabbah/Bureau/STAGE/figaro/datasets/Identif/Data_JoB/15_06_2023/LeoH/Mocap/Resampled/Trial0_Resampled.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in spamreader:
#         if 'time' in row[0]:
#             print('First')
#         else:
#             new_row=row[0].split(',')
#             F = pin.Force(np.array([float(new_row[-9]),float(new_row[-8]),float(new_row[-7]),float(new_row[-6]),float(new_row[-5]),float(new_row[-4])]))
#             F = F.se3Action(FP)
#             forces_trajectories[c,:] = np.array([-F.linear[0],F.linear[1],-F.linear[2],-F.angular[0],-F.angular[1],-F.angular[2]])
#             c+=1


# signal1 = forces_trajectories[:,2]
# df = pd.read_csv('/home/msabbah/Bureau/STAGE/figaro/datasets/Identif/Data_JoB/15_06_2023/LeoH/Xsens/Resampled/LeoH_Trial0_Segment_Acceleration_Resampled.csv')
# signal2 = df["Right Foot z"]

# print(signal1.shape,signal2.shape)

# # SIGNAL 2 HAS MORE DATA THAN SIGNAL 1 HERE

# corr = correlate(signal2,signal1)

# lags = correlation_lags(len(signal2[:len(signal1)]),len(signal1))

# lag = lags[np.argmax(corr)]

# print(lag)

# fig, axs = plt.subplots(2,1)
# axs[0].plot(signal1,label='mocap')
# axs[1].plot(signal2[abs(lag):len(signal1)+abs(lag)], label='xsens')
# axs[0].legend()
# axs[1].legend()
# plt.show()

# TIME SYNC KINEMATICS

q_mocap = np.loadtxt("/home/msabbah/Bureau/STAGE/figaro/datasets/Identif/Data_JoB/16_06_2023/Momo/Non synchro/Mocap/q_raw_Momo_19000.txt",delimiter=',')
q_Xsens = np.loadtxt("/home/msabbah/Bureau/STAGE/figaro/datasets/Identif/Data_JoB/16_06_2023/Momo/Non synchro/Xsens/Raw/q/qraw_Momo.txt",delimiter=',')

q_Xsens[:,6]=-q_Xsens[:,6]

signal1=q_mocap[:,10]
signal2= q_Xsens[:,10]

print(signal1.shape,signal2.shape)

# SIGNAL 2 HAS MORE DATA THAN SIGNAL 1 HERE

corr = correlate(signal2,signal1)

lags = correlation_lags(len(signal2[:len(signal1)]),len(signal1))

lag = lags[np.argmax(corr)]

print(lag)

# plt.plot(signal1,label='mocap')
# plt.plot(signal2[abs(lag):len(signal1)+abs(lag)], label='xsens')
# plt.legend()
# plt.show()

q_Xsens = q_Xsens[abs(lag):len(signal1)+abs(lag),:]

