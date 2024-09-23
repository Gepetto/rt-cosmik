import os
import sys
from pathlib import Path
# Get the current working directory
script_dir = Path.cwd()
# Construct the path to mpc_utils
rt_cosmik_path = script_dir.parent / 'rt-cosmik'
# Append it to sys.path
sys.path.append(str(rt_cosmik_path))

import pinocchio as pin
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from utils.model_utils import Robot

# Loading human urdf
human = Robot('models/human_urdf/urdf/human.urdf','models') 
human_model = human.model
human_data = human_model.createData()
human_visual_model = human.visual_model
human_collision_model = human.collision_model

q=pd.read_csv('output/q.csv').to_numpy()[:,-5:].astype(float)
q = q[200:1000,:]

dt = 1/30
dq = np.diff(q,axis=0)/dt
ddq = np.diff(dq,axis=0)/dt

q=q[:-2,:]
dq = dq[:-1,:]

fig, ax = plt.subplots(5,1)
ax[0].plot(q[:,0])
ax[0].set_ylabel('q1')
ax[1].plot(q[:,1])
ax[1].set_ylabel('q2')
ax[2].plot(q[:,2])
ax[2].set_ylabel('q3')
ax[3].plot(q[:,3])
ax[3].set_ylabel('q4')
ax[4].plot(q[:,4])
ax[4].set_ylabel('q5')
fig.suptitle('Joint angles estimated by RT COSMIK')
plt.show()

fig, ax = plt.subplots(5,1)
ax[0].plot(dq[:,0])
ax[0].set_ylabel('q1')
ax[1].plot(dq[:,1])
ax[1].set_ylabel('q2')
ax[2].plot(dq[:,2])
ax[2].set_ylabel('q3')
ax[3].plot(dq[:,3])
ax[3].set_ylabel('q4')
ax[4].plot(dq[:,4])
ax[4].set_ylabel('q5')
fig.suptitle('Joint velocities estimated by RT COSMIK')
plt.show()

fig, ax = plt.subplots(5,1)
ax[0].plot(ddq[:,0])
ax[0].set_ylabel('q1')
ax[1].plot(ddq[:,1])
ax[1].set_ylabel('q2')
ax[2].plot(ddq[:,2])
ax[2].set_ylabel('q3')
ax[3].plot(ddq[:,3])
ax[3].set_ylabel('q4')
ax[4].plot(ddq[:,4])
ax[4].set_ylabel('q5')
fig.suptitle('Joint accelerations estimated by RT COSMIK')
plt.show()

pos_hand = np.zeros((q.shape[0],3))
vel_hand = np.zeros((dq.shape[0],3))
acc_hand = np.zeros((ddq.shape[0],3))

for ii in range(q.shape[0]):
    pin.forwardKinematics(human_model,human_data,q[ii,:],dq[ii,:],ddq[ii,:])
    pin.updateFramePlacements(human_model,human_data)
    pos_hand[ii,:] = human_data.oMf[human_model.getFrameId('hand')].translation
    vel_hand[ii,:] = pin.getFrameVelocity(human_model,human_data,human_model.getFrameId('hand'),pin.LOCAL).angular
    acc_hand[ii,:] = pin.getFrameClassicalAcceleration(human_model,human_data,human_model.getFrameId('hand'),pin.LOCAL).linear

fig, ax = plt.subplots(3,1)
ax[0].plot(pos_hand[:,0])
ax[0].set_ylabel('x')
ax[1].plot(pos_hand[:,1])
ax[1].set_ylabel('y')
ax[2].plot(pos_hand[:,2])
ax[2].set_ylabel('z')
fig.suptitle('Hand translation estimated by RT COSMIK')
plt.show()

fig, ax = plt.subplots(3,1)
ax[0].plot(vel_hand[:,0])
ax[0].set_ylabel('x')
ax[1].plot(vel_hand[:,1])
ax[1].set_ylabel('y')
ax[2].plot(vel_hand[:,2])
ax[2].set_ylabel('z')
fig.suptitle('Hand velocity estimated by RT COSMIK')
plt.show()

fig, ax = plt.subplots(3,1)
ax[0].plot(acc_hand[:,0])
ax[0].set_ylabel('x')
ax[1].plot(acc_hand[:,1])
ax[1].set_ylabel('y')
ax[2].plot(acc_hand[:,2])
ax[2].set_ylabel('z')
fig.suptitle('Hand acceleration estimated by RT COSMIK')
plt.show()
