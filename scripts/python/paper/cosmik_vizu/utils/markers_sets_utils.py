from typing import Dict

# dictionnary to map the joint those placement should be updated and the corresponding segment name
SGTS_JOINTS_CALIB_MAPPING={
    "pelvis" :'root_joint',
    "right_upperleg":'right_hip_Z', # parent root_joint
    "right_lowerleg":'right_knee_Z',# parent right_hip_Y
    "right_foot":'right_ankle_Z',
    
    "left_upperleg":'left_hip_Z',
    "left_lowerleg":'left_knee_Z',
    'left_foot':'left_ankle_Z',
    
    # #"thorax":'middle_lumbar_Z', 
    "torso":'middle_cervical_Z',
   # "torso":'right_clavicle_joint_X',
   # "torso":'left_clavicle_joint_X',
    # #"head":'middle_cervical_Z',
    # "right_clavicle":'right_clavicle_joint_X',
    # "left_clavicle":'left_clavicle_joint_X',
    
    "right_upperarm":'right_shoulder_Z',
    "right_lowerarm":'right_elbow_Z',
    "right_hand":'right_wrist_Z',
 
    "left_upperarm":'left_shoulder_Z',
    "left_lowerarm":'left_elbow_Z',
    "left_hand":'left_wrist_Z'
 }
 
 

 
 
    # JOINT_SEGMENTS_COSMIK_DEFINITION={
        
    #     #'root_joint':'pelvis',
    #     'left_hip_Z':'left_upperleg',
    #     'left_knee':'left_lowerleg',
    #     'left_ankle_Z':'left_foot',
        
    #     'right_hip_Z':'right_upperleg',
    #     'right_knee':'right_lowerleg',
    #     'right_ankle_Z':'right_foot',
        
    #     'middle_thoracic_Z':'abdomen',
    #     'left_clavicle_joint_X':'torso',
    #     'right_clavicle_joint_X':'torso',
    #     'middle_cervical_Z':'torso',
        
    #     'right_shoulder_Z':'right_upperarm',
    #     'right_elbow_Z':'right_lowerarm',
    #     'right_wrist_Z':'right_hand',
        
    #     'left_shoulder_Z':'left_upperarm',
    #     'left_elbow_Z':'leftt_lowerarm',    
    #     'right_wrist_Z':'left_hand'
    # }
 

# dictionnary describing the names of the markers that are required to calibrate a given segment
# this dictionnary was written on 25th of June based on the model_utils get_??_pose functions
SGTS_MKS_MAPPING = {
     "head":      ['RSHO', 'LSHO', 'FHD','BHD', 'Head','RHD','LHD','REar','LEar'],   # for this segments we need to ask Mohamed the reference       

     "pelvis":    ['RPSI','LPSI','RASI','LASI'],    
    
     "torso":     ['RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI', 'C7'],
     "thorax":     ['RSHO', 'LSHO', 'RASI', 'LASI', 'RPSI', 'LPSI', 'C7'], 

     "rigth_upperarm": ['LSHO','RSHO','RMELB','RELB'],    
     "rigth_lowerarm": ['RMELB','RELB','RMWRI','RWRI'],        
    
     "left_upperarm": ['LSHO','RSHO','LMELB','LELB'],     
     "left_lowerarm": ['LMELB','LELB','LMWRI','LWRI'],        
    
            
    
     "right_upperleg":    ['RASI','LASI','RKNE','RMKNE'],                
     "right_lowerleg":    ['RKNE','RMKNE','RMANK','RANK'],             
     "right_foot":     ['RMANK','RANK','RTOE','R5MHD','RHEE'],     
    
     "left_upperleg":    ['LASI','RASI','LKNE','LMKNE'],                
     "left_lowerleg":    ['LKNE','LMKNE','LMANK','LANK'],              
     "left_foot":     ['LMANK','LANK','LTOE','L5MHD','LHEE'],     
        
    }


def get_segments_mks_dict(mks_positions)->Dict:
     
    """
    Filters the segment-to-marker mapping dictionary to retain only segments
    whose required markers are all present in the given marker positions dictionary.

    Parameters
    ----------
    SGTS_MKS_MAPPING : dict
        Dictionary mapping segment names (str) to a list of marker labels (list of str).
        Example:
        {
            "pelvis": ['RPSI', 'LPSI', 'RASI', 'LASI'],
            "left_upperleg": ['LASI', 'RASI', 'LKNE', 'LMKNE'],
            ...
        }

    mks_positions : dict
        Dictionary mapping marker labels (str) to their 3D positions (e.g., numpy arrays).
        Example:
        {
            'RASI': np.array([...]),
            'LASI': np.array([...]),
            ...
        }

    Returns
    -------
    """
    
    sgts_mks_dict = {}

    for segment, markers in SGTS_MKS_MAPPING.items():
        if all(marker in mks_positions for marker in markers):
            sgts_mks_dict[segment] = markers
    
    return sgts_mks_dict





# #for info the full list of OpenCap/COSMIK markers is 
# MKS_COSMIK=[
# "RASI",
# "LASI",
# "RPSI",
# "LPSI",
# "C7",
# "RSHO",
# "LSHO",
# "RELB",
# "LELB",
# "RMELB",
# "LMELB",
# "RWRI",
# "LWRI",
# "RMWRI",
# "LMWRI",
# "RKNE",
# "LKNE",
# "RMKNE",
# "LMKNE",
# "RANK",
# "LANK",
# "RMANK",
# "LMANK",
# "R5MHD",
# "L5MHD",
# "RTOE",
# "LTOE",
# "RHEE",
# "LHEE",
# "RTHI1",
# "RTHI2",
# "RTHI3",
# "LTHI1",
# "LTHI2",
# "LTHI3",
# "RTIB1",
# "RTIB2",
# "RTIB3",
# "LTIB1",
# "LTIB2",
# "LTIB3",
# "RHJC",
# "LHJC"]


