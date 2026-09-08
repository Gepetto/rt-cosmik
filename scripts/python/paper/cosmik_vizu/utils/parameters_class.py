import dataclasses
import numpy as np
import pinocchio as pin
from typing import List, Dict


@dataclasses.dataclass
class Parameters:
    size: float
    weigth: float
    gender: str
    nb_samples: int 
    optimal_control: bool
    cop_lim: list
    L: list
    pyf: int
    groups_joint_torques:dict
    dt: float = 0.01
    Tf: float = 0.01
    free_flyer: bool = True
    active_joints: list = dataclasses.field(default_factory=list)
    
   
    active_costs: list = dataclasses.field(default_factory=list)


    
    
    viewer: str = "meshcat"
    solver: str = "ipopt"
    external_forces: str = "optimal_forces_estimation"
    
    qdi: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    q_min: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    q_max: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    dq_lim: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    dqdi: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    n_tau: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    n_dq: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
    n_ddq: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))
 