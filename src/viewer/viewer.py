from utils.settings import Settings
settings = Settings()
if settings.viewer == 'ros':
    from .ros_viewer import ros_init, publish_keypoints_as_marker_array, publish_augmented_markers, publish_kinematics
else: # default to gepetto viewer
    from .gv_viewer import gv_init, place_objects

class Viewer:
    def __init__(self, model, geom_model, visual_model, keypoint_names, marker_names, freeflyer=False):
        self._model = model
        self._geom_model = geom_model 
        self._visual_model = visual_model
        self._keypoint_names = keypoint_names
        self._marker_names = marker_names
        self._freeflyer = freeflyer
        self._viewer_type = settings.viewer

        # Gepetto viewer specific
        self._viz = None

        # ROS specific publishers
        self._marker_pub = None
        self._keypoints_pub = None
        self._q_pub = None
        self._br = None
        
        if self._viewer_type == 'ros':
            self._keypoints_pub, self._marker_pub, self._q_pub, self._br = ros_init(self._freeflyer)
        else :
            self._viz = gv_init(self._model, self._geom_model, self._visual_model, self._keypoint_names, self._marker_names)
    
    def display_q(self, q):
        if self._viewer_type == 'ros':
            publish_kinematics(q, self._q_pub, self._model.names, self._br)
        else:
            self._viz.display(q)

    def display_keypoints(self, pos_keypoints_dict):
        if self._viewer_type == 'ros':
            publish_keypoints_as_marker_array(list(pos_keypoints_dict.values()), self._keypoints_pub, pos_keypoints_dict.keys())
        else:
            place_objects(self._viz, self._keypoint_names, pos_keypoints_dict)

    def display_markers(self, pos_markers_dict):
        if self._viewer_type == 'ros':
            publish_augmented_markers(list(pos_markers_dict.values()), self._marker_pub, pos_markers_dict.keys())
        else:
            place_objects(self._viz, self._marker_names, pos_markers_dict)