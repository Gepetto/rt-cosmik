import pinocchio as pin 
import casadi 
import pinocchio.casadi as cpin 
import quadprog
from typing import Dict, List
from collections import deque
import numpy as np 

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
    qp_G = .5 * (P + P.T) + np.eye(P.shape[0])*(1e-8)   # make sure P is symmetric, pos,def
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

class RT_IK:
    """_Class to manage multi body IK problem using qp solver quadprog_
    """
    def __init__(self,model: pin.Model, dict_m: Dict, q0: np.ndarray, keys_to_track_list: List, dt: float, dict_dof_to_keypoints=None, with_freeflyer=True) -> None:
        """_Init of the class _

        Args:
            model (pin.Model): _Pinocchio biomechanical model_
            dict_m (Dict): _a dictionnary containing the measures of the landmarks_
            q0 (np.ndarray): _initial configuration_
            keys_to_track_list (List): _name of the points to track from the dictionnary_
            dt (float): _Sampling rate of the data_
            dict_dof_to_keypoints (Dict): _a dictionnary linking frame of pinocchio model to measurements. Default to None if the pinocchio model has the same frame naming than the measurements_
            with_freeflyer (boolean): _tells if the pinocchio model has a ff or not. Default to True.
        """
        self._model = model
        self._nq = self._model.nq
        self._nv = self._model.nv
        self._data = self._model.createData()
        self._dict_m = dict_m
        self._q0 = q0
        self._dt = dt # TO SET UP : FRAMERATE OF THE DATA
        self._with_freeflyer = with_freeflyer
        self._keys_to_track_list = keys_to_track_list
        # Ensure dict_dof_to_keypoints is either a valid dictionary or None
        self._dict_dof_to_keypoints = dict_dof_to_keypoints if dict_dof_to_keypoints is not None else None

        # Casadi framework 
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()

        cq = casadi.SX.sym("q",self._nq,1)
        cdq = casadi.SX.sym("dq",self._nv,1)

        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        self._integrate = casadi.Function('integrate',[ cq,cdq ],[cpin.integrate(self._cmodel,cq,cdq) ])

        self._new_key_list = []
        cfunction_list = []
        for key in self._keys_to_track_list:
            index_mk = self._cmodel.getFrameId(key)
            if index_mk < len(self._model.frames.tolist()): # Check that the frame is in the model
                new_key = key.replace('.','')
                self._new_key_list.append(key)
                function_mk = casadi.Function(f'f_{new_key}',[cq],[self._cdata.oMf[index_mk].translation])
                cfunction_list.append(function_mk)

        self._cfunction_dict=dict(zip(self._new_key_list,cfunction_list))

        # Create a list of keys excluding the specified key
        self._keys_list = [key for key in self._dict_m.keys() if key !='Time']

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
        self._K_ii=0.5
        self._K_lim=0.75
        self._damping=1e-3
        self._max_iter = 10
        self._threshold = 0.01

        # Line search tuning 
        self._alpha = 1.0 # Start with full step size 
        self._c = 0.5 # Backtracking line search factor 
        self._beta = 0.8 # Reduction factor 

        #TODO: Change the mapping and adapt it to the model
        self._mapping_joint_angle = dict(zip(['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','L5S1_FE','L5S1_RIE','RShoulder_FE','RShoulder_AA','RShoulder_RIE','RElbow_FE','RElbow_PS','RHip_FE','RHip_AA','RHip_RIE','RKnee_FE','RAnkle_FE'],np.arange(0,self._nq,1)))


    def calculate_RMSE_dicts(self, meas:Dict, est:Dict)->float:
        """_Calculate the RMSE between a dictionnary of markers measurements and markers estimations_

        Args:
            meas (Dict): _Measured markers_
            est (Dict): _Estimated markers_

        Returns:
            float: _RMSE value for all the markers_
        """

        # Initialize lists to store all the marker positions
        all_est_pos = []
        all_meas_pos = []

        # Concatenate all marker positions and measurements
        for key in self._keys_to_track_list:
            all_est_pos.append(est[key])
            all_meas_pos.append(meas[key])

        # Convert lists to numpy arrays
        all_est_pos = np.concatenate(all_est_pos)
        all_meas_pos = np.concatenate(all_meas_pos)

        # Calculate the global RMSE
        rmse = np.sqrt(np.mean((all_meas_pos - all_est_pos) ** 2))

        return rmse
    
    def update_marker_estimates(self, q0):
        """Update the estimated marker positions."""
        pin.forwardKinematics(self._model, self._data, q0)
        pin.updateFramePlacements(self._model, self._data)

        for key in self._keys_to_track_list:
            if self._dict_dof_to_keypoints is not None:
                frame_id = self._model.getFrameId(self._dict_dof_to_keypoints.get(key))
            else:
                frame_id = self._model.getFrameId(key)
            self._dict_m_est[key] = self._data.oMf[frame_id].translation.reshape((3, 1))


    def solve_ik_sample_quadprog(self)->np.ndarray:
        """_Solve the ik optimisation problem : q* = argmin(||P_m - P_e||^2 + lambda|q_init - q|) st to q_min <= q <= q_max for a given sample _
        """

        q0=pin.normalize(self._model,self._q0)
        
        if self._with_freeflyer:
            G= np.concatenate((np.zeros((2*(self._nv-6),6)),np.concatenate((np.eye(self._nv-6),-np.eye(self._nv-6)),axis=0)),axis=1)

            Delta_q_max = (-q0[7:]+ self._model.upperPositionLimit[7:])
            Delta_q_min = (-q0[7:]+ self._model.lowerPositionLimit[7:])

        else:
            G=np.concatenate((np.eye(self._nv),-np.eye(self._nv)),axis=0) # Inequality matrix size number of inequalities (=nv) \times nv

            Delta_q_max = pin.difference(
                self._model, q0, self._model.upperPositionLimit
            )
            Delta_q_min = pin.difference(
                self._model, q0, self._model.lowerPositionLimit
            )

        p_max = self._K_lim * Delta_q_max
        p_min = self._K_lim * Delta_q_min
        h = np.hstack([p_max, -p_min])
        
        # Reset estimated markers dict 
        self.update_marker_estimates(q0)
        
        nb_iter = 0

        rmse = self.calculate_RMSE_dicts(self._dict_m,self._dict_m_est)

        while rmse > self._threshold and nb_iter<self._max_iter:
            # Set QP matrices 
            P=np.zeros((self._nv,self._nv)) # Hessian matrix size nv \times nv
            q=np.zeros((self._nv,)) # Gradient vector size nv

            pin.forwardKinematics(self._model, self._data, q0)
            pin.updateFramePlacements(self._model,self._data)

            for marker_name in self._keys_to_track_list:

                v_ii=(self._dict_m[marker_name].reshape((3,))-self._dict_m_est[marker_name].reshape((3,)))/self._dt

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

            # print('Solving ...')
            dq=quadprog_solve_qp(P,q,G,h)

            # Line search 
            initial_rmse = rmse  # Store current RMSE
            while self._alpha > 1e-5:  # Prevent alpha from becoming too small
                q_test = pin.integrate(self._model, q0, dq * self._alpha * self._dt)
                
                self.update_marker_estimates(q_test)
                new_rmse = self.calculate_RMSE_dicts(self._dict_m, self._dict_m_est)
                
                if new_rmse < initial_rmse - self._c * self._alpha * np.dot(q.T, dq):  # Sufficient decrease condition
                    break  # Sufficient improvement found
                
                self._alpha *= self._beta  # Reduce the step size

            q0 = pin.integrate(self._model, q0, dq * self._alpha * self._dt)

            # Reset estimated markers dict 
            self.update_marker_estimates(q0)
            rmse = self.calculate_RMSE_dicts(self._dict_m,self._dict_m_est)
            nb_iter+=1

        return q0
    
    def solve_ik_sample_casadi(self)->np.ndarray:
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

        for key in self._cfunction_dict.keys():
            cost+=1*casadi.sumsqr(self._dict_m[key]-self._cfunction_dict[key](Q))

        # Set the constraint for the joint limits
        if self._with_freeflyer:
            for i in range(7,self._nq):
                opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],Q[i],self._model.upperPositionLimit[i]))
        else : 
            for i in range(self._nq):
                opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],Q[i],self._model.upperPositionLimit[i]))
        
        opti.minimize(cost)

        # Set Ipopt options to suppress output
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 50,
            "ipopt.linear_solver": "mumps",
            "expand": True
        }

        opti.solver("ipopt", opts)

        sol = opti.solve()
        
        q = sol.value(Q)
        return q 

class RT_SWIKA:
    """_Class to manage multi body Sliding Window IK problem using fatrop solver_
    """
    def __init__(self,model: pin.Model, deque_dict_m: deque, x_list: List, q_list: List, keys_to_track_list: List, T: int, dt: float, dict_dof_to_keypoints=None, with_freeflyer=True) -> None:
       
        """ _Init of the class _

        Args:
            model (pin.Model): _Pinocchio biomechanical model_
            deque_dict_m (deque): _a deque containing the measures of the landmarks_
            x_list (List): _list of states_
            q_list (List): _list of joint angles (include the freeflyer if there is one)_
            keys_to_track_list (List): _name of the points to track from the dictionnary_
            T (int): _Size of the window_
            dt (float): _Sampling rate of the data_
            dict_dof_to_keypoints (Dict): _a dictionnary linking frame of pinocchio model to measurements. Default to None if the pinocchio model has the same frame naming than the measurements_
            with_freeflyer (boolean): _tells if the pinocchio model has a ff or not. Default to True.
        """
        
        self._model = model
        self._nq = self._model.nq
        self._nv = self._model.nv
        self._data = self._model.createData()
        self._deque_dict_m = deque_dict_m
        self._x_list = x_list
        self._q_list = q_list
        self._keys_to_track_list = keys_to_track_list
        self._T = T
        self._dt = dt # TO SET UP : FRAMERATE OF THE DATA
        # Ensure dict_dof_to_keypoints is either a valid dictionary or None
        self._dict_dof_to_keypoints = dict_dof_to_keypoints if dict_dof_to_keypoints is not None else None
        self._with_freeflyer = with_freeflyer

        # Casadi framework 
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()

        q = casadi.SX.sym("q",self._nq) # q
        Dq = casadi.SX.sym("Dq", self._nv) # variation of q 
        dq = casadi.SX.sym("dq",self._nv) # dq
        ddq = casadi.SX.sym("ddq",self._nv) # ddq
        cdt = casadi.SX.sym("dt")

        # integrate_q(q,dq) = pin.integrate(q,dq)
        self._integrate = casadi.Function('integrate',[ q,Dq ],[cpin.integrate(self._cmodel,q,Dq) ])
        cpin.framesForwardKinematics(self._cmodel, self._cdata, q)

        # States
        x = casadi.vertcat(Dq,dq)
        # Controls
        u = ddq

        # ODE rhs
        ode = casadi.vertcat(x[self._nv:],u)

        # Discretize system
        sys = {}
        sys["x"] = x
        sys["u"] = u
        sys["p"] = cdt
        sys["ode"] = ode*cdt # Time scaling

        intg = casadi.integrator('intg','rk',sys,0,1,{"simplify":True, "number_of_finite_elements": 4})
        self._Fdyn = casadi.Function('Fdyn',[x,u,cdt],[intg(x0=x,u=u,p=cdt)["xf"]],["x","u","dt"],["xnext"])

        self._nx = x.numel()
        self._nu = u.numel()

        self._new_key_list = []
        cfunction_list = []
        for key in self._keys_to_track_list:
            index_mk = self._cmodel.getFrameId(key)
            if index_mk < len(self._model.frames.tolist()): # Check that the frame is in the model
                new_key = key.replace('.','')
                self._new_key_list.append(key)
                function_mk = casadi.Function(f'f_{new_key}',[q],[self._cdata.oMf[index_mk].translation])
                cfunction_list.append(function_mk)

        self._cfunction_dict=dict(zip(self._new_key_list,cfunction_list))

    def solve_swika_casadi(self)->np.ndarray:
        x_list = self._x_list
        q_list = self._q_list
        lstm_dict_list = list(self._deque_dict_m)
        
        # Casadi optimization class
        opti = casadi.Opti()

        Q = []
        X = []
        U = []

        for k in range(self._T):
            X.append(opti.variable(self._nx))
            Q.append(self._integrate(q_list[k],X[k][:self._nv]))
            U.append(opti.variable(self._nu))

        X0 = x_list

        cost = 0

        for k in range(self._T):
            # Markers tracking cost function
            for key in self._cfunction_dict.keys():
                cost+=1*casadi.sumsqr(lstm_dict_list[k][key]-self._cfunction_dict[key](Q[k]))

            # Control regul
            cost += 1e-5*casadi.sumsqr(U[k])

            # State regul
            cost += 1e-3*casadi.sumsqr(X[k]-X0[k])

            if k != self._T-1:
                # Multiple shooting gap-closing constraint
                opti.subject_to(X[k+1]==self._Fdyn(X[k],U[k],self._dt))
                
            # joint limits 
            # Set the constraint for the joint limits
            if self._with_freeflyer:
                for i in range(6,self._nv):
                    opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],X[k][i],self._model.upperPositionLimit[i]))
            else : 
                for i in range(self._nv):
                    opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],X[k][i],self._model.upperPositionLimit[i]))
                
            opti.set_initial(X[k],X0[k])

        X = casadi.hcat(X)
        
        opti.minimize(cost)

        options = {}
        options["expand"] = True
        options["fatrop"] = {"mu_init": 0.01}
        options["fatrop"]={"max_iter":50}
        options["fatrop"]={"tol":1e-3}
        options["structure_detection"] = "auto"
        options["debug"] = False
        
        # (codegen of helper functions)
        # options["jit"] = True
        # options["jit_temp_suffix"] = False
        # options["jit_options"] = {"flags": ["-O3"],"compiler": "ccache gcc"}

        opti.solver("fatrop",options)

        sol = opti.solve()

        new_X = sol.value(X).T
        solved_x_list = [new_X[i] for i in range(new_X.shape[0])] 

        solved_q_list = [pin.integrate(self._model, q_list[i],solved_x_list[i][:self._nv]) for i in range(len(q_list))] 

        return sol, solved_x_list, solved_q_list #X with Dq but also Q with ff quat 