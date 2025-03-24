import sys
import os

cosmik_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.insert(0, cosmik_path) # Repo root
sys.path.insert(0, os.path.join(cosmik_path, "src")) # src dir

import numpy as np
from rtcosmik.ergonomics.reba import RebaScore


sample_pose = np.array([[ 0.08533354,  1.03611605,  0.09013124],
                    [ 0.15391247,  0.91162637, -0.00353906],
                    [ 0.22379057,  0.87361878,  0.11541229],
                    [ 0.4084777 ,  0.69462843,  0.1775224 ],
                    [ 0.31665226,  0.46389668,  0.16556387],
                    [ 0.1239769 ,  0.82994377, -0.11715403],
                    [ 0.08302169,  0.58146328, -0.19830338],
                    [-0.06767788,  0.53928527, -0.00511249],
                    [ 0.11368726,  0.49372503,  0.21275574],
                    [ 0.069179  ,  0.07140968,  0.26841402],
                    [ 0.10831762, -0.36339359,  0.34032449],
                    [ 0.11368726,  0.41275504, -0.01171348],
                    [ 0.        ,  0.        ,  0.        ],
                    [ 0.02535541, -0.43954643,  0.04373671],
                    [ 0.26709431,  0.33643749,  0.17985192],
                    [-0.15117603,  0.49462711,  0.02703403]])

rebaScore = RebaScore()

body_params = rebaScore.get_body_angles_from_pose_right(sample_pose)
arms_params = rebaScore.get_arms_angles_from_pose_right(sample_pose)

rebaScore.set_body(body_params)
score_a, partial_a = rebaScore.compute_score_a()

rebaScore.set_arms(arms_params)
score_b, partial_b = rebaScore.compute_score_b()

score_c, caption = rebaScore.compute_score_c(score_a, score_b)

print("Score A: ", score_a, "Partial: ", partial_a)
print("Score B: ", score_b, "Partial: ", partial_b)
print("Score C: ", score_c, caption)