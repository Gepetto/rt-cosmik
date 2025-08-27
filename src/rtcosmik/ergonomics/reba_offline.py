import numpy as np
import pandas as pd

class RebaScore:
    '''
    Class to compute REBA metrics
    '''
    def __init__(self, angles_csv_path, positions_csv_path, frame_number, 
                legs_monopodal=False, load_weight=0, hard_conditions=False,
                right_upper_arm_leaning=False, left_upper_arm_leaning=False,
                easiness_of_grabbing_score=0, static_longer_than_1_minute=False,
                More_than_4_times_repeated_per_minute=False, Highly_dynamic_or_instable=False,
                scaled_disance_rshoulder_to_rASIS=0, scaled_disance_lshoulder_to_lASIS=0):

        self.angles_csv_path = angles_csv_path
        self.positions_csv_path = positions_csv_path
        self.frame_number = frame_number

        # Step 1 params
        self.neck = {"neck_angle_z": self.get_joint_angle_by_name("Cervical_flex_ext"),
                    "neck_lateral_flexion": self.get_joint_angle_by_name("Cervical_lat_bend"), 
                    "neck_axial_rotation": self.get_joint_angle_by_name("Cervical_int_ext_rot")}

        # Step 2 params
        self.trunk = {"trunk_angle_z": self.get_joint_angle_by_name("Lumbar_flex_ext"), 
                    "trunk_lateral_flexion": self.get_joint_angle_by_name("Lumbar_lateral_flex"), 
                    "trunk_axial_rotation": 0}#self.get_joint_angle_by_name("thoracic_rot_int_ext")}

        # Step 3 params
        self.legs = {"legs_monopodal": legs_monopodal,
                    "right_leg_angle_z": self.get_joint_angle_by_name("Rankle_flex_ext"), 
                    "left_leg_angle_z": self.get_joint_angle_by_name("Lankle_flex_ext")} 

        # Step 4 params
        self.init_table_a()

        # Step 5 params
        self.load = {"load_weight": load_weight, 
                    "hard_conditions": hard_conditions}

        # Step 6 params
        self.init_table_c()

        # Step 7 params
        self.upper_arms = {"right_upper_arm_angle_z": self.get_joint_angle_by_name("Rshoulder_flex_ext"), 
                            "left_upper_arm_angle_z": self.get_joint_angle_by_name("Lshoulder_flex_ext"), 
                            "right_shoulder_distance_to_rASIS": np.linalg.norm(self.get_joint_position_by_name("r_shoulder_study")-self.get_joint_position_by_name("r.ASIS_study")),
                            "left_shoulder_distance_to_lASIS": np.linalg.norm(self.get_joint_position_by_name("L_shoulder_study")-self.get_joint_position_by_name("L.ASIS_study")),
                            "right_upper_arm_abduction": self.get_joint_angle_by_name("Rshoulder_abd_add"), 
                            "left_upper_arm_abduction": self.get_joint_angle_by_name("Lshoulder_abd_add"),
                            "right_upper_arm_leaning": right_upper_arm_leaning, 
                            "left_upper_arm_leaning": left_upper_arm_leaning}
        self.scaled_disance_rshoulder_to_rASIS = scaled_disance_rshoulder_to_rASIS
        self.scaled_disance_lshoulder_to_lASIS = scaled_disance_lshoulder_to_lASIS
        
        # Step 8 params
        self.lower_arms = {"right_lower_arm_angle_z": self.get_joint_angle_by_name("Relbow_flex_ext"), 
                            "left_lower_arm_angle_z": self.get_joint_angle_by_name("Lelbow_flex_ext")}

        # Step 9 params
        self.wrists = {"right_wrist_angle_z": 0,#self.get_joint_angle_by_name("Rwrist_flex_ext"), 
                        "left_wrist_angle_z": 0,#self.get_joint_angle_by_name("Lwrist_flex_ext"),
                        "right_wrist_angle_x": 0,#self.get_joint_angle_by_name("Rwrist_x"), 
                        "left_wrist_angle_x": 0}#self.get_joint_angle_by_name("Lwrist_x")}
            
        # Step 10 params
        self.init_table_b()

        # Step 11 params
        self.easiness_of_grabbing = {"easiness_of_grabbing_score": easiness_of_grabbing_score}

        # Step 12 params
        # Table C already initialized

        # Step 13 params
        # Table C already initialized

        # Step 14 params
        self.pose_context = {"static_longer_than_1_minute": static_longer_than_1_minute, 
                            "More_than_4_times_repeated_per_minute": More_than_4_times_repeated_per_minute, 
                            "Highly_dynamic_or_instable": Highly_dynamic_or_instable}

    def init_table_a(self):
        '''
        Table used to compute upper body score

        :return: None
        '''
        self.table_a = np.array([
                                [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
                                [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
                                [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]]
                                ])

    def init_table_b(self):
        '''
        Table used to computer lower body score

        :return: None
        '''
        self.table_b = np.array([
                                [[1, 2, 2], [1, 2, 3]],
                                [[1, 2, 3], [2, 3, 4]],
                                [[3, 4, 5], [4, 5, 5]],
                                [[4, 5, 5], [5, 6, 7]],
                                [[6, 7, 8], [7, 8, 8]],
                                [[7, 8, 8], [8, 9, 9]],
                                ])

    def init_table_c(self):
        '''
        Table to compute score_c

        :return: None
        '''
        self.table_c = np.array([
                                [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
                                [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
                                [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
                                [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
                                [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
                                [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
                                [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
                                [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
                                [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
                                [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
                                [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
                                [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                                ])


    def compute_score_a(self):


        # Step 1
        score_step1 = 1
        if self.neck["neck_angle_z"] > 20 or self.neck["neck_angle_z"] < 0:
            score_step1 += 1
        if abs(self.neck["neck_lateral_flexion"]) > 15 or abs(self.neck["neck_axial_rotation"]) > 15:
            score_step1 += 1
        # print("score step 1 :", score_step1)

        # Step 2
        score_step2 = 1
        if (self.trunk["trunk_angle_z"] < -5 and self.trunk["trunk_angle_z"] > -20) or (
            self.trunk["trunk_angle_z"] > 5 and self.trunk["trunk_angle_z"] < 20):
            score_step2 += 1
        elif (self.trunk["trunk_angle_z"] >= 20 and self.trunk["trunk_angle_z"] < 60) or self.trunk["trunk_angle_z"] < -20:
            score_step2 += 2
        elif self.trunk["trunk_angle_z"] >= 60:
            score_step2 += 3
        if abs(self.trunk["trunk_lateral_flexion"]) > 5 or abs(self.trunk["trunk_axial_rotation"]) > 5:
            score_step2 += 1
        # print("score step 2 :", score_step2)


        # Step 3
        # Step 3 right
        score_step3_right = 1
        if self.legs["legs_monopodal"]:
            score_step3_right += 1
        if self.legs["right_leg_angle_z"] > 30 and self.legs["right_leg_angle_z"] < 60:
            score_step3_right += 1
        elif self.legs["right_leg_angle_z"] >= 60:
            score_step3_right += 2
        # print("score step 3 right :", score_step3_right)
        # Step 3 left
        score_step3_left = 1
        if self.legs["legs_monopodal"]:
            score_step3_left += 1
        if self.legs["left_leg_angle_z"] > 30 and self.legs["left_leg_angle_z"] < 60:
            score_step3_left += 1
        elif self.legs["left_leg_angle_z"] >= 60:
            score_step3_left += 2
        # print("score step 3 left :", score_step3_left)
        score_step3 = np.max([score_step3_right, score_step3_left])
        # print("score step 3 :", score_step3)

        # Step 4
        score_a = self.table_a[score_step1-1, score_step2-1, score_step3-1]

        return score_a
        

    def compute_score_b(self):

        # Step 7
        # Step 7 right
        score_step7_right = 1
        if (self.upper_arms["right_upper_arm_angle_z"] < -20 or (
            self.upper_arms["right_upper_arm_angle_z"] > 20 and self.upper_arms["right_upper_arm_angle_z"] < 45)):
            score_step7_right += 1
        elif self.upper_arms["right_upper_arm_angle_z"] >= 45 and self.upper_arms["right_upper_arm_angle_z"] < 90:
            score_step7_right += 2
        elif self.upper_arms["right_upper_arm_angle_z"] >= 90:
            score_step7_right += 3
        if self.upper_arms["right_shoulder_distance_to_rASIS"] > self.scaled_disance_rshoulder_to_rASIS + 0.025:
            score_step7_right += 1
        if self.upper_arms["right_upper_arm_abduction"] > 15:
            score_step7_right += 1
        if self.upper_arms["right_upper_arm_leaning"]:
            score_step7_right -= 1
        # print("score step 7 right :", score_step7_right)
        # Step 7 left
        score_step7_left = 1
        if (self.upper_arms["left_upper_arm_angle_z"] < -20 or (
            self.upper_arms["left_upper_arm_angle_z"] > 20 and self.upper_arms["left_upper_arm_angle_z"] < 45)):
            score_step7_left += 1
        elif self.upper_arms["left_upper_arm_angle_z"] >= 45 and self.upper_arms["left_upper_arm_angle_z"] < 90:
            score_step7_left += 2
        elif self.upper_arms["left_upper_arm_angle_z"] >= 90:
            score_step7_left += 3
        if self.upper_arms["left_shoulder_distance_to_lASIS"] > self.scaled_disance_lshoulder_to_lASIS + 0.025:
            score_step7_left += 1
        if self.upper_arms["left_upper_arm_abduction"] > 15:
            score_step7_left += 1
        if self.upper_arms["left_upper_arm_leaning"]:
            score_step7_left -= 1
        # print("score step 7 left :", score_step7_left)
        score_step7 = np.max([score_step7_right, score_step7_left])
        # print("score step 7 :", score_step7)

        # Step 8
        # Step 8 right
        score_step8_right = 1
        if self.lower_arms["right_lower_arm_angle_z"] < 60 or self.lower_arms["right_lower_arm_angle_z"] > 100:
            score_step8_right += 1
        # print("score step 8 right :", score_step8_right)
        # Step 8 left
        score_step8_left = 1
        if self.lower_arms["left_lower_arm_angle_z"] < 60 or self.lower_arms["left_lower_arm_angle_z"] > 100:
            score_step8_left += 1
        # print("score step 8 left :", score_step8_left)
        score_step8 = np.max([score_step8_right, score_step8_left])
        # print("score step 8 :", score_step8)
    
        # Step 9
        # Step 9 right
        score_step9_right = 1
        if abs(self.wrists["right_wrist_angle_z"]) > 15:
            score_step9_right += 1
        if abs(self.wrists["right_wrist_angle_x"]) > 15:
            score_step9_right += 1
        # print("score step 9 right :", score_step9_right)
        # Step 9 left
        score_step9_left = 1
        if abs(self.wrists["left_wrist_angle_z"]) > 15:
            score_step9_left += 1
        if abs(self.wrists["left_wrist_angle_x"]) > 15:
            score_step9_left += 1
        # print("score step 9 left :", score_step9_left)
        score_step9 = np.max([score_step9_right, score_step9_left])
        # print("score step 9 :", score_step9)

        # Step 10
        score_b = self.table_b[score_step7-1, score_step8-1, score_step9-1]

        return score_b

    def compute_score_c(self, score_a, score_b):

        # Step 11
        score_step11 = self.easiness_of_grabbing["easiness_of_grabbing_score"]
        # print("score step 11 :", score_step11)
        
        # Step 12
        score_step12 = score_b + score_step11
        # print("score step 12 :", score_step12)

        # Step 5
        score_step5 = 0
        if self.load["load_weight"] >= 5 and self.load["load_weight"] < 10:
            score_step5 += 1
        elif self.load["load_weight"] >= 10:
            score_step5 += 2
        if self.load["hard_conditions"]:
            score_step5 += 1
        # print("score step 5 :", score_step5)

        # Step 6
        score_step6 = score_a + score_step5
        # print("score step 6 :", score_step6)

        # Step 13
        score_step13 = self.table_c[score_step6-1, score_step12-1]
        # print("score step 13 :", score_step13)

        score_c = score_step13

        return score_c


    def compute_reba_score(self):

        self.score_a = self.compute_score_a()
        # print("score a :", self.score_a)

        self.score_b = self.compute_score_b()
        # print("score b :", self.score_b)

        self.score_c = self.compute_score_c(self.score_a, self.score_b)
        # print("score c :", self.score_c)

        # Step 14
        score_step14 = 0
        if self.pose_context["static_longer_than_1_minute"]:
            score_step14 += 1
        if self.pose_context["More_than_4_times_repeated_per_minute"]:
            score_step14 += 1
        if self.pose_context["Highly_dynamic_or_instable"]:
            score_step14 += 1
        # print("score step 14 :", score_step14)

        score_reba = self.score_c + score_step14
        # print("score reba :", score_reba)

        return score_reba



    def get_joint_angle_by_name(self, angle_name):

        angles_data = pd.read_csv(self.angles_csv_path)
        angle = np.rad2deg(float(angles_data[angle_name].values[self.frame_number]))

        return angle


    def get_joint_position_by_name(self, joint_name):
        
        positions_data = pd.read_csv(self.positions_csv_path)
        position_x = positions_data[joint_name + '_x'].values[self.frame_number]
        position_y = positions_data[joint_name + '_y'].values[self.frame_number]
        position_z = positions_data[joint_name + '_z'].values[self.frame_number]
        joint_postion = np.array([position_x, position_y, position_z], dtype=np.float32)

        return joint_postion