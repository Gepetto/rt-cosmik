from pinocchio.visualize import GepettoVisualizer
import pinocchio as pin 
import numpy as np 
import sys

def gv_init(model, geom_model, visual_model, keypoint_names=None, marker_names=None):
    viz = GepettoVisualizer(model, geom_model,visual_model)
    try:
        viz.initViewer()
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install gepetto-viewer"
        )
        print(err)
        sys.exit(0)

    try:
        viz.loadViewerModel("pinocchio")
    except AttributeError as err:
        print(
            "Error while loading the viewer model. It seems you should start gepetto-viewer"
        )
        print(err)
        sys.exit(0)
    
    #add world frame
    viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.02, 0.15)
    place(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))

    # Init objects to show 
    # Frame axis for frame in the pinocchio model 
    for frame in model.frames.tolist():
        viz.viewer.gui.addXYZaxis('world/'+frame.name,[1,0,0,1],0.01,0.1)
    
    # Keypoints
    if keypoint_names is not None : 
        for keypoint in keypoint_names:
            viz.viewer.gui.addSphere('world/'+keypoint,0.01,[0,1,0,1])

    # measured markers from mocap or lstm
    if marker_names is not None: 
        for marker in marker_names:
            viz.viewer.gui.addSphere('world/'+marker,0.01,[0,0,1,1])

    return viz

#add frame 
def add_frames(viz,seg_frames,w,r,s):
    for seg_name, mks in seg_frames.items():
        frame_name = f'world/{seg_name+"_"+w}'
        viz.viewer.gui.addXYZaxis(frame_name, [255, 0., 0, 1.], r, s)


#add model markers
def add_marker(viz,marker_names, r, g,b):
    for marker in marker_names:
        viz.viewer.gui.addSphere('world/'+marker+"_m",0.01,[r,g,b,1])

def Rquat(x, y, z, w):
    q = pin.Quaternion(x, y, z, w)
    q.normalize()
    return q.matrix()  

def place(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    viz.viewer.gui.refresh()

def place_objects(viz, objects_pos_dict):
    for name in objects_pos_dict.keys():
        M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([objects_pos_dict[name][0],objects_pos_dict[name][1],objects_pos_dict[name][2]]).T))
        place(viz, 'world/'+name, M)

