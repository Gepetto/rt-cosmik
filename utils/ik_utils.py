import pinocchio as pin 
import casadi 
import pinocchio.casadi as cpin 
import quadprog
import cyipopt
from scipy.optimize import approx_fprime
from typing import Dict, List
import numpy as np 
from scipy.spatial.transform import Rotation as R

def quadprog_solve_qp(P: np.ndarray, q: np.ndarray, G: np.ndarray=None, h: np.ndarray=None, A: np.ndarray=None, b: np.ndarray=None):
    """_Set up the qp solver using quadprog API_

    Args:
        P (np.ndarray): _Hessian matrix of the qp_
        q (np.ndarray): _Gradient vector of the qp_
        G (np.ndarray, optional): _Inequality constraints matrix_. Defaults to None.
        h (np.ndarray, optional): _Vector for inequality constraints_. Defaults to None.
        A (np.ndarray, optional): _Equality constraints matrix_. Defaults to None.
        b (np.ndarray, optional): _Vector for equality constraints_. Defaults to None.

    Returns:
        _launch solve_qp of quadprog solver_
    """
    qp_G = .5 * (P + P.T) #+ np.eye(P.shape[0])*(1e-5)   # make sure P is symmetric, pos,def
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0] 

class Ipopt_warm_start(object):

    def __init__(self,  model,meas,keys,dict_dof_to_keypoints,with_freeflyer):
        self._meas=meas
        self._keys=keys
        self._with_freeflyer=with_freeflyer
        self._dict_dof_to_keypoints=dict_dof_to_keypoints
        self._model=model
        self._data=self._model.createData()
        
    def objective(self, x):
        # callback for objective 
        pin.forwardKinematics(self._model, self._data, x)
        pin.updateFramePlacements(self._model,self._data) 
      
        Goal=np.empty(shape=[0,3])
        markers_pos=[]

        for key in self._keys:
            Goal=np.concatenate((Goal,np.reshape(np.array(self._meas[key]),(1,3))),axis=0)
            if self._dict_dof_to_keypoints is not None :
                markers_pos.append(self._data.oMf[self._model.getFrameId(self._dict_dof_to_keypoints[key])].translation)
            else :
                markers_pos.append(self._data.oMf[self._model.getFrameId(key)].translation)           
        
        J=np.sum((Goal-markers_pos)**2)

        return  J 

    def constraints(self, x):
        """Returns the constraints."""
        if self._with_freeflyer:
            return np.linalg.norm([x[3],x[4],x[5],x[6]]) # norm of the freeflyer quaternion equal to 1
        else : 
            return 1

    def gradient(self, x):
        # callback for gradient

        G=approx_fprime(x, self.objective, 1e-5)

        return G

    def jacobian(self, x):
        # callback for jacobian of constraints
        jac=approx_fprime(x, self.constraints, 1e-5)

        return jac

class IK_Quadprog:
    """_Class to manage multi body IK problem using qp solver quadprog_
    """
    def __init__(self,model: pin.Model, dict_m: Dict, q0: np.ndarray, keys_to_track_list: List, dt: float, dict_dof_to_keypoints=None, with_freeflyer=True) -> None:
        """_Init of the class _

        Args:
            model (pin.Model): _Pinocchio biomechanical model_
            dict_m (Dict): _a dictionnary containing the measures of the landmarks_
            q0 (np.ndarray): _initial configuration_
            keys_to_track_list (List): _name of the points to track from the dictionnary_
            dict_dof_to_keypoints (Dict): _a dictionnary linking frame of pinocchio model to measurements. Default to None if the pinocchio model has the same frame naming than the measurements_
            with_freeflyer (boolean): _tells if the pinocchio model has a ff or not. Default to True.
        """
        self._model = model
        self._data = self._model.createData()
        self._dict_m = dict_m
        self._q0 = q0
        self._with_freeflyer = with_freeflyer

        self._dt = dt # TO SET UP : FRAMERATE OF THE DATA

        # Create a list of keys excluding the specified key
        self._keys_list = [key for key in self._dict_m[0].keys() if key !='Time']

        self._nq = self._model.nq
        self._nv = self._model.nv

        self._keys_to_track_list = keys_to_track_list
        # Ensure dict_dof_to_keypoints is either a valid dictionary or None
        self._dict_dof_to_keypoints = dict_dof_to_keypoints if dict_dof_to_keypoints is not None else None

        pin.forwardKinematics(self._model, self._data, self._q0)
        pin.updateFramePlacements(self._model, self._data)

        markers_est_pos = []
        if self._dict_dof_to_keypoints:
            # If a mapping dictionary is provided, use it
            for key in self._keys_to_track_list:
                frame_id = self._dict_dof_to_keypoints.get(key)
                if frame_id:
                    markers_est_pos.append(self._data.oMf[self._model.getFrameId(frame_id)].translation.reshape((3, 1)))
        else:
            # Direct linking with Pinocchio model frames
            for key in self._keys_to_track_list:
                markers_est_pos.append(self._data.oMf[self._model.getFrameId(key)].translation.reshape((3, 1)))

        self._dict_m_est = dict(zip(self._keys_to_track_list, markers_est_pos))

        # Quadprog and qp settings
        self._K_ii=1
        self._K_lim=1
        self._damping=1e-3

    def create_meas_list(self)-> List[Dict]:
        """_Create a list with each element is a dictionnary of measurements referencing a given sample_

        Returns:
            List[Dict]: _List of dictionnary of measures_
        """
        d_list = []
        for i in range(len(self._dict_m)):
            meas_i = []
            for key in self._keys_list:
                meas_i.append(self._dict_m[i][key])
            d_i = dict(zip(self._keys_list,meas_i))
            d_list.append(d_i)
        return d_list
    
    # def calculate_RMSE_dicts(self, meas:Dict, est:Dict)->float:
    #     """_Calculate the RMSE between a dictionnary of markers measurements and markers estimations_

    #     Args:
    #         meas (Dict): _Measured markers_
    #         est (Dict): _Estimated markers_

    #     Returns:
    #         float: _RMSE value for all the markers_
    #     """

    #     # Initialize lists to store all the marker positions
    #     all_est_pos = []
    #     all_meas_pos = []

    #     # Concatenate all marker positions and measurements
    #     for key in self._keys_to_track_list:
    #         all_est_pos.append(est[key])
    #         all_meas_pos.append(meas[key])

    #     # Convert lists to numpy arrays
    #     all_est_pos = np.concatenate(all_est_pos)
    #     all_meas_pos = np.concatenate(all_meas_pos)

    #     # Calculate the global RMSE
    #     rmse = np.sqrt(np.mean((all_meas_pos - all_est_pos) ** 2))

    #     return rmse
    
    def solve_ik_sample(self, ii: int, meas: Dict)->None:
        """_Solve the ik optimisation problem : q* = argmin(||P_m - P_e||^2 + lambda|q_init - q|) st to q_min <= q <= q_max for a given sample _

        Args:
            ii (int): _number of sample_
            meas (Dict): _Dictionnary of landmark measurements_

        """
        if ii == 0 : # Init to be done with ipopt
            lb = self._model.lowerPositionLimit # lower joint limits
            ub = self._model.upperPositionLimit # upper joint limits
            cl=cu=[1]

            nlp = cyipopt.Problem(
                n=len(self._q0),
                m=len(cl),
                problem_obj=Ipopt_warm_start(self._model,meas,self._keys_to_track_list,self._dict_dof_to_keypoints, self._with_freeflyer),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu,
                )

            nlp.add_option('tol',1e-3)
            nlp.add_option('print_level',0)
            q_opt, info = nlp.solve(self._q0)

            return q_opt

        else : # QP running
            q0=self._q0
            
            # Reset estimated markers dict 
            pin.forwardKinematics(self._model, self._data, q0)
            pin.updateFramePlacements(self._model,self._data)
            
            markers_est_pos = []
            if self._dict_dof_to_keypoints:
                # If a mapping dictionary is provided, use it
                for key in self._keys_to_track_list:
                    frame_id = self._dict_dof_to_keypoints.get(key)
                    if frame_id:
                        markers_est_pos.append(self._data.oMf[self._model.getFrameId(frame_id)].translation.reshape((3, 1)))
            else:
                # Direct linking with Pinocchio model frames
                for key in self._keys_to_track_list:
                    markers_est_pos.append(self._data.oMf[self._model.getFrameId(key)].translation.reshape((3, 1)))

            self._dict_m_est = dict(zip(self._keys_to_track_list, markers_est_pos))

            if self._with_freeflyer:
                # Set QP matrices 
                P=np.zeros((self._nv,self._nv)) # Hessian matrix size nv \times nv
                q=np.zeros((self._nv,)) # Gradient vector size nv
                G=np.concatenate((np.zeros((2*(self._nv-6),6)),np.concatenate((np.identity(self._nv-6),-np.identity(self._nv-6)),axis=0)),axis=1) # Inequality matrix size number of inequalities (=nv) \times nv

                q_max_n=self._K_lim*(self._model.upperPositionLimit[7:]-q0[7:])/self._dt
                q_min_n=self._K_lim*(-self._model.lowerPositionLimit[7:]+q0[7:])/self._dt
                h=np.reshape((np.concatenate((q_max_n,q_min_n),axis=0)),(2*len(q_max_n),))

                pin.forwardKinematics(self._model, self._data, q0)
                pin.updateFramePlacements(self._model,self._data)
                for marker_name in self._keys_to_track_list:

                    if self._dict_dof_to_keypoints is not None :
                        self._dict_m_est[marker_name]=self._data.oMf[self._model.getFrameId(self._dict_dof_to_keypoints[marker_name])].translation.reshape((3,1))
                    else :
                        self._dict_m_est[marker_name]=self._data.oMf[self._model.getFrameId(marker_name)].translation.reshape((3,1))

                    v_ii=(meas[marker_name]-self._dict_m_est[marker_name])/self._dt
                    mu_ii=self._damping*np.dot(v_ii.T,v_ii)
                    
                    if self._dict_dof_to_keypoints is not None :
                        J_ii=pin.computeFrameJacobian(self._model,self._data,q0,self._model.getFrameId(self._dict_dof_to_keypoints[marker_name]),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    else :
                        J_ii=pin.computeFrameJacobian(self._model,self._data,q0,self._model.getFrameId(marker_name),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

                    P_ii=np.matmul(J_ii_reduced.T,J_ii_reduced)+mu_ii*np.eye(self._nv)
                    P+=P_ii

                    q_ii=np.matmul(-self._K_ii*v_ii.T,J_ii_reduced)
                    q+=q_ii.flatten()

                print('Solving for ' + str(ii) +'...')
                dq=quadprog_solve_qp(P,q,G,h)
                q0=pin.integrate(self._model,q0,dq*self._dt)

            else :
                # Set QP matrices 
                P=np.zeros((self._nv,self._nv)) # Hessian matrix size nv \times nv
                q=np.zeros((self._nv,)) # Gradient vector size nv
                G=np.concatenate((np.identity(self._nv),-np.identity(self._nv)),axis=0) # Inequality matrix size number of inequalities (=nv) \times nv

                q_max_n=self._K_lim*(self._model.upperPositionLimit-q0)/self._dt
                q_min_n=self._K_lim*(-self._model.lowerPositionLimit+q0)/self._dt
                h=np.reshape((np.concatenate((q_max_n,q_min_n),axis=0)),(2*len(q_max_n),))

                pin.forwardKinematics(self._model, self._data, q0)
                pin.updateFramePlacements(self._model,self._data)
                for marker_name in self._keys_to_track_list:

                    if self._dict_dof_to_keypoints is not None :
                        self._dict_m_est[marker_name]=self._data.oMf[self._model.getFrameId(self._dict_dof_to_keypoints[marker_name])].translation.reshape((3,1))
                    else :
                        self._dict_m_est[marker_name]=self._data.oMf[self._model.getFrameId(marker_name)].translation.reshape((3,1))

                    v_ii=(meas[marker_name]-self._dict_m_est[marker_name])/self._dt

                    mu_ii=self._damping*np.dot(v_ii.T,v_ii)

                    
                    if self._dict_dof_to_keypoints is not None :
                        J_ii=pin.computeFrameJacobian(self._model,self._data,q0,self._model.getFrameId(self._dict_dof_to_keypoints[marker_name]),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    else :
                        J_ii=pin.computeFrameJacobian(self._model,self._data,q0,self._model.getFrameId(marker_name),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    
                    J_ii_reduced=J_ii[:3,:]

                    P_ii=np.matmul(J_ii_reduced.T,J_ii_reduced)+mu_ii*np.eye(self._nv)
                    P+=P_ii

                    q_ii=np.matmul(-self._K_ii*v_ii.T,J_ii_reduced)
                    q+=q_ii.flatten()

                print('Solving for ' + str(ii) +'...')
                dq=quadprog_solve_qp(P,q,G,h)
                q0=pin.integrate(self._model,q0,dq*self._dt)


            return q0

    
    def solve_ik(self)->List:
        """_Returns a list of joint angle configuration over the whole trajectory _

        Returns:
            List: _List of q_i_
        """
        q_list = []
        meas_list = self.create_meas_list()
        for i in range(len(meas_list)):
            self._q0=self.solve_ik_sample(i, meas_list[i])
            q_list.append(self._q0)
        return q_list
        

class IK_Casadi:
    """ Class to manage multi body IK problem using pinocchio casadi 
    """
    def __init__(self,model: pin.Model, dict_m: Dict, q0: np.ndarray):
        """_Init of the class _

        Args:
            model (pin.Model): _Pinocchio biomechanical model_
            dict_m (Dict): _a dictionnary containing the measures of the landmarks_
            q0 (np.ndarray): _initial configuration_
        """
        self._model = model
        self._dict_m = dict_m
        self._q0 = q0
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()


        # Create a list of keys excluding the specified key
        self._keys_list = [key for key in self._dict_m[0].keys() if key !='Time']

        ### CASADI FRAMEWORK
        self._nq = self._cmodel.nq
        self._nv = self._cmodel.nv

        cq = casadi.SX.sym("q",self._nq,1)
        cdq = casadi.SX.sym("dq",self._nv,1)

        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        self._integrate = casadi.Function('integrate',[ cq,cdq ],[cpin.integrate(self._cmodel,cq,cdq) ])

        cfunction_list = []
        self._new_key_list = [] # Only take the frames that are in the model 


        self._keys_to_track_list = ['LIAS', 'RIAS', 'LIPS', 'RIPS', 'LFLE', 'LFAL', 'LFCC', 'RFLE', 'RFAL', 'RFCC', 'LFME', 'LTAM', 'LFM5', 'LFM1', 'RFM5', 'RFM1', 'RTAM', 'RFME']

        for key in self._keys_to_track_list:
            index_mk = self._cmodel.getFrameId(key)
            if index_mk < len(self._model.frames.tolist()): # Check that the frame is in the model
                new_key = key.replace('.','')
                self._new_key_list.append(key)
                function_mk = casadi.Function(f'f_{new_key}',[cq],[self._cdata.oMf[index_mk].translation])
                cfunction_list.append(function_mk)

        self._cfunction_dict=dict(zip(self._new_key_list,cfunction_list))

        self._mapping_joint_angle = dict(zip(['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','Hip_Z_R', 'Hip_X_R', 'Hip_Y_R', 'Knee_Z_R', 'Ankle_Z_R', 'Ankle_X_R','Hip_Z_L','Hip_X_L', 'Hip_Y_L', 'Knee_Z_L', 'Ankle_Z_L', 'Ankle_X_L'],np.arange(0,self._nq,1)))

    def create_meas_list(self)-> List[Dict]:
        """_Create a list with each element is a dictionnary of measurements referencing a given sample_

        Returns:
            List[Dict]: _List of dictionnary of measures_
        """
        d_list = []
        for i in range(len(self._dict_m)):
            meas_i = []
            for key in self._keys_list:
                meas_i.append(self._dict_m[i][key])
            d_i = dict(zip(self._keys_list,meas_i))
            d_list.append(d_i)
        return d_list

    def solve_ik_sample(self, ii: int, meas: Dict)->np.ndarray:
        """_Solve the ik optimisation problem : q* = argmin(||P_m - P_e||^2 + lambda|q_init - q|) st to q_min <= q <= q_max for a given sample _

        Args:
            ii (int): _number of sample_
            meas (Dict): _Dictionnary of landmark measurements_

        Returns:
            np.ndarray: _q_i joint angle at the i-th sample_
        """

        joint_to_regularize = [] #['RElbow_FE','RElbow_PS','RHip_RIE']
        value_to_regul = 0.001

        # Casadi optimization class
        opti = casadi.Opti()

        # Variables MX type
        DQ = opti.variable(self._nv)
        Q = self._integrate(self._q0,DQ)

        omega = 1e-6*np.ones(self._nq)

        for name in joint_to_regularize :
            if name in self._mapping_joint_angle:
                omega[self._mapping_joint_angle[name]] = value_to_regul # Adapt the weight for given joints, for instance the hip Y
            else :
                raise ValueError("Joint to regulate not in the model")

        cost = 0

        if ii == 0:
            for key in self._cfunction_dict.keys():
                cost+=1*casadi.sumsqr(meas[key]-self._cfunction_dict[key](Q))
        else : 
            for key in self._cfunction_dict.keys():
                cost+=1*casadi.sumsqr(meas[key]-self._cfunction_dict[key](Q))  + 0.001*casadi.sumsqr(casadi.dot(omega,self._q0-Q))

        # Set the constraint for the joint limits
        for i in range(7,self._nq):
            opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],Q[i],self._model.upperPositionLimit[i]))
        
        opti.minimize(cost)

        # Set Ipopt options to suppress output
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 1000,
            "ipopt.linear_solver": "mumps"
        }

        opti.solver("ipopt", opts)

        print('Solving for ' + str(ii) +'...')
        sol = opti.solve()
        
        q_i = sol.value(Q)
        return q_i 

    def solve_ik(self)->List:
        """_Returns a list of joint angle configuration over the whole trajectory _

        Returns:
            List: _List of q_i_
        """
        q_list = []
        meas_list = self.create_meas_list()
        for i in range(len(meas_list)):
            self._q0 = self.solve_ik_sample(i, meas_list[i])
            q_list.append(self._q0)
        return q_list

class IK_Singlebody:
    """ Class to manage a single IK problem using pinocchio casadi 
    """
    def __init__(self):
        pass

    def rotation_matrix_to_euler(self, R: np.ndarray, convention: str)->np.ndarray:
        r = R.from_matrix(R)
        return r.as_euler(convention)

    def quaternion_to_euler(self, q: np.ndarray, convention: str)->np.ndarray:
        r = R.from_quat(q)
        return r.as_euler(convention)

    def rodrigues_to_euler(self, r_vec: np.ndarray, convention: str)->np.ndarray:
        theta = np.linalg.norm(r_vec)
        if theta == 0:
            return np.zeros(3)
        axis = r_vec / theta
        r = R.from_rotvec(axis * theta)
        return r.as_euler(convention)

    def to_rotation_matrix(self, orientation: np.ndarray)->np.ndarray:
        if isinstance(orientation, np.ndarray) and orientation.shape == (3, 3):
            return orientation
        elif isinstance(orientation, np.ndarray) and orientation.shape == (4,):
            return R.from_quat(orientation).as_matrix()
        elif isinstance(orientation, np.ndarray) and orientation.shape == (3,):
            return R.from_rotvec(orientation).as_matrix()
        else:
            raise ValueError("Invalid orientation format")

    def solve_ik(self, parent_orientation: np.ndarray, child_orientation: np.ndarray, convention: str):
        # Convert parent_orientation and child_orientation to rotation matrices if they are not already
        parent_matrix = self.to_rotation_matrix(parent_orientation)
        child_matrix = self.to_rotation_matrix(child_orientation)

        # Calculate the relative rotation matrix
        relative_matrix = np.linalg.inv(parent_matrix).dot(child_matrix)

        # Convert the relative rotation matrix to euler angles with the given convention
        joint_angles = self.rotation_matrix_to_euler(relative_matrix, convention)
        
        return joint_angles

# # Example usage
# dk = IK_Singlebody()
# parent_orientation_matrix = np.eye(3)
# child_orientation_matrix = R.from_euler('xyz', [45, 45, 45], degrees=True).as_matrix()
# convention = 'xyz'
# joint_angles = dk.solve_ik(parent_orientation_matrix, child_orientation_matrix, convention)
# print("Joint angles:", joint_angles)





