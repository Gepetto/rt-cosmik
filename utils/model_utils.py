from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np 

class Robot(RobotWrapper):
    """_Class to load a given urdf_

    Args:
        RobotWrapper (_type_): _description_
    """
    def __init__(self,robot_urdf,package_dirs,isFext=False,freeflyer_ori =None,):
        """_Init of the robot class. User can choose between floating base or not and to set the transformation matrix for this floating base._

        Args:
            robot_urdf (_str_): _path to the robot urdf_
            package_dirs (_str_): _path to the meshes_
            isFext (bool, optional): _Adds a floating base if set to True_. Defaults to False.
            freeflyer_ori (_array_, optional): _Orientation of the floating base, given as a rotation matrix_. Defaults to None.
        """

        # intrinsic dynamic parameter names
        self.params_name = (
            "Ixx",
            "Ixy",
            "Ixz",
            "Iyy",
            "Iyz",
            "Izz",
            "mx",
            "my",
            "mz",
            "m",
        )

        # defining conditions
        self.isFext = isFext

        # folder location
        self.robot_urdf = robot_urdf

        # initializing robot's models
        if not isFext:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs)
        else:
            self.initFromURDF(robot_urdf, package_dirs=package_dirs,
                              root_joint=pin.JointModelFreeFlyer())
            
        if freeflyer_ori is not None and isFext == True : 
            self.model.jointPlacements[self.model.getJointId('root_joint')].rotation = freeflyer_ori
            ub = self.model.upperPositionLimit
            ub[:7] = 1
            self.model.upperPositionLimit = ub
            lb = self.model.lowerPositionLimit
            lb[:7] = -1
            self.model.lowerPositionLimit = lb
            self.data = self.model.createData()

        ## \todo test that this is equivalent to reloading the model
        self.geom_model = self.collision_model
    
def model_scaling_df(model, keypoints_df):
    lowerleg_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Ankle')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Ankle')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Ankle')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['Z']]))
    upperleg_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Knee')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['Z']]))
    trunk_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Hip')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['Z']]))
    upperarm_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Shoulder')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['Z']]))
    lowerarm_l = np.linalg.norm(np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Elbow')]['Z']])-np.array([keypoints_df[(keypoints_df['Keypoint'] == 'Right Wrist')]['X'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Wrist')]['Y'],keypoints_df[(keypoints_df['Keypoint'] == 'Right Wrist')]['Z']]))

    model.jointPlacements[model.getJointId('knee_Z')].translation=np.array([lowerleg_l,0,0])
    model.jointPlacements[model.getJointId('lumbar_Z')].translation=np.array([upperleg_l,0,0])
    model.jointPlacements[model.getJointId('shoulder_Z')].translation=np.array([trunk_l,0,0])
    model.jointPlacements[model.getJointId('elbow_Z')].translation=np.array([upperarm_l,0,0])
    model.frames[model.getFrameId('hand_fixed')].translation=np.array([lowerarm_l,0,0])
    model.frames[model.getFrameId('hand')].translation=np.array([lowerarm_l,0,0])

    data=model.createData()

    return model, data

def model_scaling(model, keypoints):

    keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]


    mapping = dict(zip(keypoint_names,[i for i in range(len(keypoint_names))]))

    lowerleg_l = np.linalg.norm(keypoints[mapping['Right Ankle']][:]-keypoints[mapping['Right Knee']][:])
    upperleg_l = np.linalg.norm(keypoints[mapping['Right Knee']][:]-keypoints[mapping['Right Hip']][:])    
    trunk_l = np.linalg.norm(keypoints[mapping['Right Hip']][:]-keypoints[mapping['Right Shoulder']][:])
    upperarm_l = np.linalg.norm(keypoints[mapping['Right Shoulder']][:]-keypoints[mapping['Right Elbow']][:])
    lowerarm_l = np.linalg.norm(keypoints[mapping['Right Elbow']][:]-keypoints[mapping['Right Wrist']][:])

    model.jointPlacements[model.getJointId('knee_Z')].translation=np.array([lowerleg_l,0,0])
    model.jointPlacements[model.getJointId('lumbar_Z')].translation=np.array([upperleg_l,0,0])
    model.jointPlacements[model.getJointId('shoulder_Z')].translation=np.array([trunk_l,0,0])
    model.jointPlacements[model.getJointId('elbow_Z')].translation=np.array([upperarm_l,0,0])
    model.frames[model.getFrameId('hand_fixed')].translation=np.array([lowerarm_l,0,0])
    model.frames[model.getFrameId('hand')].translation=np.array([lowerarm_l,0,0])

    data=model.createData()

    return model, data

