import pinocchio as pin
import numpy as np
import time 

def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

def visualize_joint_angle_results(directory_name:str, viz, model):
    dofs_names = ['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','L5S1_FE','RShoulder_FE','RShoulder_AA','RShoulder_RIE','RElbow_FE','RElbow_PS','RHip_FE','RHip_AA','RKnee_FE','RAnkle_FE']
    q=[]
    for name in dofs_names: 
        q_i = np.loadtxt(directory_name+'/'+name+'.csv')
        q.append(q_i)
    
    q=np.array(q)

    input('Are you ready to record the results ?')
    for ii in range(q.shape[1]):
        q_ii=q[:,ii]
        print(q_ii)
        viz.display(q_ii)
        time.sleep(0.016)


def visualize_model_and_measurements(model: pin.Model, q: np.ndarray, lstm_mks_dict: dict, seg_names_mks: dict, sleep_time: float, viz):
    """    _Function to visualize model markers from q's and raw lstm markers from lstm_
    """
    data = model.createData()

    viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
    
    for seg_name, mks in seg_names_mks.items():
        viz.viewer.gui.addXYZaxis(f'world/{seg_name}', [255, 0., 0, 1.], 0.008, 0.08)
        for mk_name in mks:
            sphere_name_model = f'world/{mk_name}_model'
            sphere_name_raw = f'world/{mk_name}_raw'
            viz.viewer.gui.addSphere(sphere_name_model, 0.01, [0, 0., 255, 1.])
            viz.viewer.gui.addSphere(sphere_name_raw, 0.01, [255, 0., 0, 1.])

    for i in range(q.shape[1]):
        pin.forwardKinematics(model, data, q[:,i])
        pin.updateFramePlacements(model, data)
        viz.display(q[:,i])
        for seg_name, mks in seg_names_mks.items():
            #Display markers from model
            for mk_name in mks:
                sphere_name_model = f'world/{mk_name}_model'
                sphere_name_raw = f'world/{mk_name}_raw'
                mk_position = data.oMf[model.getFrameId(mk_name)].translation
                place(viz, sphere_name_model, pin.SE3(np.eye(3), np.matrix(mk_position.reshape(3,)).T))
                place(viz, sphere_name_raw, pin.SE3(np.eye(3), np.matrix(lstm_mks_dict[i][mk_name].reshape(3,)).T))
            
            #Display frames from model
            frame_name = f'world/{seg_name}'
            frame_se3= data.oMf[model.getFrameId(seg_name)]
            place(viz, frame_name, frame_se3)
        time.sleep(sleep_time)


def vizualise_triangulated_landmarks(jcps_dict: dict, nb_frames: int, sleep_time: float, viz):
    """    _Function to visualize model markers from q's and raw lstm markers from lstm_
    """
    viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
    
    for name, pos in jcps_dict.items():
        sphere_name = f'world/{name}_model'
        viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 0., 255, 1.])

    for i in range(nb_frames):
        for name, pos in jcps_dict.items():
            sphere_name = f'world/{name}_model'
            place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(jcps_dict[name][i].reshape(3,)).T))
        time.sleep(sleep_time)

# def vizualise_triangulated_landmarks_and_lstm(jcps_dict: dict, nb_frames: int, sleep_time: float, viz):
#     """    _Function to visualize model markers from q's and raw lstm markers from lstm_
#     """

#     viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.04, 0.2)
    
#     for name, pos in jcps_dict.items():
#         sphere_name = f'world/{name}_model'
#         viz.viewer.gui.addSphere(sphere_name, 0.01, [0, 0., 255, 1.])

#     for i in range(nb_frames):
#         for name, pos in jcps_dict.items():
#             sphere_name = f'world/{name}_model'
#             print(sphere_name)
#             print(jcps_dict[name][i].reshape(3,))
#             place(viz, sphere_name, pin.SE3(np.eye(3), np.matrix(jcps_dict[name][i].reshape(3,)).T))
#         time.sleep(sleep_time)