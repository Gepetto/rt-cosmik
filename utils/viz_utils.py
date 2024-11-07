import pinocchio as pin
import numpy as np
import time 
# import cv2

def Rquat(x, y, z, w):
    q = pin.Quaternion(x, y, z, w)
    q.normalize()
    return q.matrix()  

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


### MMPOSE VISUALISATION
VISUALIZATION_CFG = dict(
    body26=dict(
        skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 2), (5, 18),(6, 18), (17,18), # Head, shoulders, and neck connections
                    (5, 7), (7, 9),                                                              # Right arm connections
                    (6, 8), (8, 10),                                                             # Left arm connections
                    (18, 19),                                                                    # Trunk connection
                    (11, 13), (13, 15), (15, 20), (15, 22), (15, 24),                            # Left leg and foot connections
                    (12, 14), (14, 16), (16, 21), (16, 23), (16, 25),                            # Right leg and foot connections
                    (12, 19), (11, 19)],                                                         # Hip connection

        # Updated palette
        palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]],
    
        # Updated link color
        link_color = [
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
            2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
            2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],

        # Updated point color
        point_color = [
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,
            5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5,
            5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas = [0.026] * 26
    ),
    coco=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        palette=[(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                 (255, 153, 255), (153, 204, 255), (255, 102, 255),
                 (255, 51, 255), (102, 178, 255), (51, 153, 255),
                 (255, 153, 153), (255, 102, 102), (255, 51, 51),
                 (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                 (0, 0, 255), (255, 0, 0), (255, 255, 255)],
        link_color=[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ],
        point_color=[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]),
    coco_wholebody=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                  (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                  (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                  (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                  (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                  (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                  (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                  (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                  (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                  (129, 130), (130, 131), (131, 132)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255),
                 (255, 153, 255), (102, 178, 255), (255, 51, 51)],
        link_color=[
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
            1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1,
            1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
            0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
            0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
            0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
            0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
            0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
            0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
            0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
            0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
            0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
            0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
        ]))

def visualize(frame,
              results,
              output_dir,
              idx,
              frame_id,
              thr=0.5,
              resize=1280,
              skeleton_type='coco'):

    skeleton = VISUALIZATION_CFG[skeleton_type]['skeleton']
    palette = VISUALIZATION_CFG[skeleton_type]['palette']
    link_color = VISUALIZATION_CFG[skeleton_type]['link_color']
    point_color = VISUALIZATION_CFG[skeleton_type]['point_color']

    scale = resize / max(frame.shape[0], frame.shape[1])
    keypoints, bboxes, _ = results
    scores = keypoints[..., 2]
    keypoints = (keypoints[..., :2] * scale).astype(int)
    bboxes *= scale
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    for kpts, score, bbox in zip(keypoints, scores, bboxes):
        show = [1] * len(kpts)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                         cv2.LINE_AA)
            else:
                show[u] = show[v] = 0
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
                cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)
    if output_dir:
        cv2.imwrite(f'{output_dir}/{str(frame_id).zfill(6)}.jpg', img)
    else:
        cv2.imshow('pose_tracker'+str(idx), img)
        return cv2.waitKey(1) != 'q'
    return True