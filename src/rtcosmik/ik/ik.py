import pinocchio as pin 
import casadi 
import pinocchio.casadi as cpin 
import quadprog
from typing import Dict, List
import numpy as np 
import time
from os import system

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
    def __init__(self,model: pin.Model, dict_m: Dict, q0: np.ndarray, keys_to_track_list: List, dt: float, omega: Dict, dict_dof_to_keypoints=None, with_freeflyer=True) -> None:
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
        # Reverse keys and values
        self._dict_keypoints_to_dof = {value: key for key, value in dict_dof_to_keypoints.items()} if dict_dof_to_keypoints is not None else None

        # Casadi framework 
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()

        cq = casadi.SX.sym("q",self._nq,1)
        cdq = casadi.SX.sym("dq",self._nv,1)

        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        self._integrate = casadi.Function('integrate',[ cq,cdq ],[cpin.integrate(self._cmodel,cq,cdq) ])

        self._new_key_list = []
        cfunction_list = []

        if self._dict_dof_to_keypoints:
            for key in self._keys_to_track_list:
                index_mk = self._cmodel.getFrameId(key)
                if index_mk >= len(self._model.frames.tolist()): # Check that the frame is in the model
                    new_index_mk = self._cmodel.getFrameId(self._dict_keypoints_to_dof[key])
                    new_key = self._dict_keypoints_to_dof[key].replace('.','')
                    self._new_key_list.append(new_key)
                    function_mk = casadi.Function(f'f_{new_key}',[cq],[self._cdata.oMf[new_index_mk].translation])
                    cfunction_list.append(function_mk)
                elif index_mk < len(self._model.frames.tolist()): # Check that the frame is in the model
                    new_key = key.replace('.','')
                    self._new_key_list.append(key)
                    function_mk = casadi.Function(f'f_{new_key}',[cq],[self._cdata.oMf[index_mk].translation])
                    cfunction_list.append(function_mk)
        else:
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
        self._max_iter = 3
        self._threshold = 0.01

        # Line search tuning 
        self._alpha = 1.0 # Start with full step size 
        self._c = 0.5 # Backtracking line search factor 
        self._beta = 0.8 # Reduction factor 

        # #TODO: Change the mapping and adapt it to the model
        # self._mapping_joint_angle = dict(zip(['FF_TX','FF_TY','FF_TZ','FF_Rquat0','FF_Rquat1','FF_Rquat2','FF_Rquat3','L5S1_FE','L5S1_RIE','RShoulder_FE','RShoulder_AA','RShoulder_RIE','RElbow_FE','RElbow_PS','RHip_FE','RHip_AA','RHip_RIE','RKnee_FE','RAnkle_FE'],np.arange(0,self._nq,1)))
        self.omega = omega

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
            if self._dict_keypoints_to_dof is not None:
                frame_id = self._model.getFrameId(self._dict_keypoints_to_dof[key])
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

                if self._dict_keypoints_to_dof is not None :
                    J_ii=pin.computeFrameJacobian(self._model,self._data,q0,self._model.getFrameId(self._dict_keypoints_to_dof[marker_name]),pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
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
    
    # def solve_ik_sample_casadi(self) -> np.ndarray:
    #     # Parameters
    #     joint_to_regularize = []  # List of joints to regularize (e.g., ['RElbow_FE', 'RHip_RIE'])
    #     value_to_regul = 0.001  # Regularization value for specific joints

    #     # CasADi optimization class
    #     opti = casadi.Opti()

    #     # Variables (MX type)
    #     DQ = opti.variable(self._nv)  # Differential joint angles
    #     Q = self._integrate(self._q0, DQ)  # Integrated joint angles

    #     # Regularization weights
    #     omega = 1e-6 * np.ones(self._nq)  # Default regularization weight
    #     for name in joint_to_regularize:
    #         if name in self._mapping_joint_angle:
    #             omega[self._mapping_joint_angle[name]] = value_to_regul  # Set specific weights
    #         else:
    #             raise ValueError(f"Joint '{name}' not found in the model")

    #     # Precompute constant terms for the cost function
    #     precomputed_terms = {}
    #     for key in self._cfunction_dict.keys():
    #         if self._dict_dof_to_keypoints:
    #             precomputed_terms[key] = self._dict_m[self._dict_dof_to_keypoints[key]]  # Use keypoints if available
    #         else:
    #             precomputed_terms[key] = self._dict_m[key]  # Directly use the measurement

    #     # Cost function (symbolic evaluation)
    #     cost = 0
    #     for key, func in self._cfunction_dict.items():
    #         res = func(Q)  # Evaluate the function symbolically
    #         cost += 1 * casadi.sumsqr(precomputed_terms[key] - res)  # Add to cost

    #     # Joint limits constraints
    #     if self._with_freeflyer:
    #         for i in range(7, self._nq):  # Skip freeflyer joints (first 7)
    #             opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i], Q[i], self._model.upperPositionLimit[i]))
    #         opti.subject_to(casadi.sumsqr(Q[3:7]) == 1)  # Quaternion normalization constraint
    #     else:
    #         for i in range(self._nq):  # Apply constraints to all joints
    #             opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i], Q[i], self._model.upperPositionLimit[i]))

    #     # Minimize the cost function
    #     opti.minimize(cost)

    #     # Solver options
    #     opts = {
    #         "ipopt.print_level": 0,  # Suppress solver output
    #         "ipopt.sb": "yes",  # Suppress banner
    #         "ipopt.max_iter": 100,  # Maximum iterations
    #         "ipopt.linear_solver": "mumps",  # Linear solver
    #         "print_time": 1,  # Print timing information
    #         "expand": True,  # Expand expressions for better performance
    #         # "ipopt.hessian_approximation": "limited-memory",  # Hessian approximation
    #         "ipopt.tol": 1e-3,  # Overall tolerance
    #         "ipopt.constr_viol_tol": 1e-6,  # Constraint violation tolerance
    #         "ipopt.compl_inf_tol": 1e-6,  # Complementarity tolerance
    #         "ipopt.dual_inf_tol": 1e-6,  # Dual infeasibility tolerance
    #         "ipopt.acceptable_tol": 1e-3,  # Acceptable tolerance
    #         "ipopt.acceptable_constr_viol_tol": 1e-5  # Acceptable constraint violation tolerance
    #     }

    #     # Solve the optimization problem
    #     opti.solver("ipopt", opts)
    #     t0 =time.time()
    #     sol = opti.solve()
    #     t1 = time.time()
    #     print("Time for opti.solve : ", t1 - t0)
        
    #     # Get the optimized joint angles
    #     q = sol.value(Q)

    #     return q
    def solve_ik_sample_casadi(self)->np.ndarray:
        # joint_to_regularize = [] #['RElbow_FE','RElbow_PS','RHip_RIE']
        # value_to_regul = 0.001
        # Casadi optimization class
        opti = casadi.Opti()

        # Variables MX type
        DQ = opti.variable(self._nv)
        Q = self._integrate(self._q0,DQ)

        # omega = 1e-6*np.ones(self._nq)

        # for name in joint_to_regularize :
        #     if name in self._mapping_joint_angle:
        #         omega[self._mapping_joint_angle[name]] = value_to_regul # Adapt the weight for given joints, for instance the hip Y
        #     else :
        #         raise ValueError("Joint to regulate not in the model")

        cost = 0

        if self._dict_dof_to_keypoints:
            for key in self._cfunction_dict.keys():
                cost+=self.omega[key]*casadi.sumsqr(self._dict_m[self._dict_dof_to_keypoints[key]]-self._cfunction_dict[key](Q))

        else:
            for key in self._cfunction_dict.keys():
                cost+=self.omega[key]*casadi.sumsqr(self._dict_m[key]-self._cfunction_dict[key](Q))

        # Set the constraint for the joint limits
        if self._with_freeflyer:
            for i in range(7,self._nq):
                opti.subject_to(opti.bounded(self._model.lowerPositionLimit[i],Q[i],self._model.upperPositionLimit[i]))
                opti.subject_to(casadi.sumsqr(Q[3:7])==1)
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
            "print_time":0,
            "expand": True,

            # Tolerance options
            "ipopt.tol": 1e-1,  # Overall tolerance for the optimization problem
            "ipopt.constr_viol_tol": 1e-6,  # Constraint violation tolerance
            "ipopt.compl_inf_tol": 1e-6,  # Complementarity tolerance
            "ipopt.dual_inf_tol": 1e-6,  # Dual infeasibility tolerance
            "ipopt.acceptable_tol": 1e-3,  # Less strict tolerance for acceptable solutions
            "ipopt.acceptable_constr_viol_tol": 1e-5  # Acceptable constraint violation tolerance
        }

        opti.solver("ipopt", opts)
        try:
            sol = opti.solve()
            q = sol.value(Q)
        except:
            q= opti.debug.value(Q)
        
        
        # Get the optimized joint angles

        # Get the optimized joint angles
        
        return q

    
class RT_SWIKA:
    def __init__(self, pin_model: pin.Model, keys_to_track: List, N: int, dict_dof_to_keypoints: Dict=None, with_freeflyer=True, code: str ='c'):
        # Initialize the Pinocchio model
        self._pin_model = pin_model
        self._nq = self._pin_model.nq
        self._nv = self._pin_model.nv
        self._nx = self._nq + self._nv
        self._nu = self._nv
        self._with_freeflyer = with_freeflyer
        self._code = code 

        self._N = N

        self._keys_to_track = keys_to_track

        # Ensure dict_dof_to_keypoints is either a valid dictionary or None
        self._dict_dof_to_keypoints = dict_dof_to_keypoints if dict_dof_to_keypoints is not None else None

        self._ocp_func = self.create_ocp()
    
    def create_ocp(self):
        ##### CASADI SYMBOLICS #####
        cmodel = cpin.Model(self._pin_model)
        cdata = cmodel.createData()

        cx = casadi.SX.sym('cx', self._nq+self._nv) # States
        cu = casadi.SX.sym('cu', self._nv) # Controls
        cdt = casadi.SX.sym('cdt') # Time step

        ### Define the constraints functions
        ## Define the dynamics function
        # Define the integrate function
        integrate = casadi.Function('integrate', [cx, cdt], [cpin.integrate(cmodel, cx[:self._nq], cx[self._nq:]*cdt)])
        # Euler integration
        qnext=integrate(cx,cdt)
        dqnext=cx[self._nq:]+cu*cdt
        xnext = casadi.vertcat(qnext,dqnext)
        dyn_fun = casadi.Function('dyn', [cx, cu, cdt], [xnext])

        ### Define the cost function
        ## Define the markers_est function
        # Perform forward kinematics
        cpin.framesForwardKinematics(cmodel, cdata, cx[:self._nq])
        # Initialize the markers_est list
        markers_est = []
        # Get the frame indices for the keys to track
        frame_indices = [cmodel.getFrameId(key) for key in self._keys_to_track]
        # Extract the translation of each frame and concatenate
        for index_mk in frame_indices:
            if index_mk < len(self._pin_model.frames.tolist()):  # Check that the frame is in the model
                markers_est = casadi.horzcat(markers_est, cdata.oMf[index_mk].translation)  # Concatenate the markers positions, size (3 x Nb of markers)
        # Create a CasADi function for the estimated markers
        fmarkers_est = casadi.Function('markers_est', [cx], [casadi.reshape(markers_est, len(self._keys_to_track) * 3, 1)])  # reorganize the markers as [x0, y0, z0, ..., xi, yi, zi, ..., xN, yN, zN]^T, size (3*Nb x 1 of markers)

        ##### OPTI FRAMEWORK #####
        opti = casadi.Opti()

        ### Define ocp parameters input
        # Time parameters
        dt = opti.parameter()

        # Measure parameter
        marker_meas = opti.parameter(len(self._keys_to_track)*3, self._N)

        # Cost parameters
        X0 = opti.parameter(self._nx)
        cost_weights =  opti.parameter(3)

        X = []
        U = []

        for k in range(self._N):
            X.append(opti.variable(self._nx))
            U.append(opti.variable(self._nu))

        # Constraints 
        for k in range(self._N):
            if k != self._N-1:
                # Euler integration
                xkp1 = dyn_fun(X[k],U[k],dt)

                # Multiple shooting gap-closing constraint
                opti.subject_to(X[k+1]==xkp1)
                
            # Set the constraint for the joint limits
            if self._with_freeflyer:
                for i in range(7,self._nq):
                    opti.subject_to(opti.bounded(self._pin_model.lowerPositionLimit[i],X[k][i],self._pin_model.upperPositionLimit[i]))
            else : 
                for i in range(self._nq):
                    opti.subject_to(opti.bounded(self._pin_model.lowerPositionLimit[i],X[k][i],self._pin_model.upperPositionLimit[i]))

        X = casadi.hcat(X)
        U = casadi.hcat(U)
        
        # Cost function 
        cost = 0
        # Markers tracking
        cost+=cost_weights[0]*casadi.sumsqr(marker_meas-fmarkers_est.map(self._N)(X))
        # State regul
        cost += cost_weights[1]*casadi.sumsqr(X-X0)
        # Control regul
        cost += cost_weights[2]*casadi.sumsqr(U)
        
        opti.minimize(cost)

        ### Define the solver
        options = {}
        options["verbose_init"] = False
        options["verbose"] = False
        options["print_time"] = False
        options["expand"] = True
        options["fatrop"] = {"print_level":0, "max_iter":50, "mu_init": 1e-5, 'warm_start_mult_bound_push' : 1e-7, "bound_push":1e-7, "tol":1e-1, "linsol_iterative_refinement":False}#, "warm_start_init_point":True}
        options["structure_detection"] = "auto"
        options["debug"] = False

        opti.solver('fatrop', options)

        ocp_func = opti.to_function('ocp', [X,U,marker_meas, X0, cost_weights, dt], [X,U], ['Xin','Uin', 'marker_meas', 'X0', 'cost_weights', 'dt'],['Xout','Uout'])
        return ocp_func
        
    def compile_Ccode(self):
        cname = self._ocp_func.generate('ocp.c', {"with_header": False, "main":True})
        oname_O3 = 'ocp_O3.so'
        print('Compiling with O3 optimization: ', oname_O3)
        t1 = time.time()
        system('gcc -fPIC -shared -O3 ' + cname + ' -o ' + oname_O3 + ' -lfatrop -lblasfeo -lm')
        t2 = time.time()
        print('Compilation time = ', (t2-t1), ' s')

    def solve(self, X: np.ndarray, U: np.ndarray, marker_meas: np.ndarray, X0: np.ndarray, cost_weights: np.ndarray, dt: float):
        if self._code == 'c': # Use codegen 
            ocp_fun = casadi.external('ocp','./ocp_O3.so')
        elif self._code == 'python': 
            ocp_fun = self._ocp_func
        else : 
            raise ValueError('Code should be either c or python')

        # print(X.shape, U.shape, marker_meas.shape, X0.shape, cost_weights.shape, dt)
        # print(ocp_fun)

        X, U = ocp_fun(X, U, marker_meas, X0, cost_weights, dt)
        return X, U