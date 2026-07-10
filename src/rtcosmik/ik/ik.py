import pinocchio as pin
import casadi
import pinocchio.casadi as cpin
import quadprog
from typing import Dict, List
import numpy as np
import time
import os
from os import system
import statistics
import collections
import ctypes

# acados is an optional backend: keep ik.py importable (e.g. for the fatrop path)
# even when acados_template is not installed. A clear error is raised only if the
# acados backend is actually selected (see RT_SWIKA_ACADOS).
try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError:
    AcadosModel = AcadosOcp = AcadosOcpSolver = None


def _read_cpu_freq_khz(core_id: int):
    """Reads the current CPU clock frequency for a given core ID from sysfs.

    Returns the frequency in kHz, or None if sysfs is restricted or unavailable.
    """
    try:
        path = f"/sys/devices/system/cpu/cpu{core_id}/cpufreq/scaling_cur_freq"
        with open(path, "r") as f:
            return float(f.read().strip())
    except Exception:
        return None


def _get_current_cpu_core() -> int:
    """Safely retrieves the current CPU core ID under Linux environments/Docker."""
    if hasattr(os, 'sched_getcpu'):
        try:
            return os.sched_getcpu()
        except Exception:
            pass
    try:
        libc = ctypes.CDLL("libc.so.6")
        res = libc.sched_getcpu()
        if res != -1:
            return res
    except Exception:
        pass
    try:
        with open("/proc/self/stat", "r") as f:
            stat_line = f.read().strip()
        fields = stat_line.rsplit(")", 1)[1].split()
        return int(fields[36])
    except Exception:
        pass
    return 0


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
        # Casadi optimization class
        opti = casadi.Opti()

        # Variables MX type
        DQ = opti.variable(self._nv)
        Q = self._integrate(self._q0,DQ)

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
        
        return q

class RT_SWIKA_FATROP:
    def __init__(self, pin_model: pin.Model, keys_to_track: List, N: int, dict_dof_to_keypoints: Dict=None, with_freeflyer=True, code: str ='c', max_iter: int = None):
        # Initialize the Pinocchio model
        self._pin_model = pin_model
        self._nq = self._pin_model.nq
        self._nv = self._pin_model.nv
        self._nx = self._nq + self._nv
        self._nu = self._nv
        self._with_freeflyer = with_freeflyer
        self._code = code 
        self._max_iter = max_iter  # shared MHE knob (settings.mhe_max_iter): None=fatrop default; caps fatrop iterations

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
        options["fatrop"] = {"print_level":0, "mu_init": 1e-1, "tol":1e-4}
        if self._max_iter is not None:
            options["fatrop"]["max_iter"] = self._max_iter
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

        X, U = ocp_fun(X, U, marker_meas, X0, cost_weights, dt)
        return X, U


class RT_SWIKA_ACADOS:
    """Acados backend for the sliding-window MHE IK, reproducing RT_SWIKA_FATROP exactly.

    This solves the *same* optimization problem as :class:`RT_SWIKA_FATROP` (the validated
    fatrop reference); acados is used purely as a faster, code-generated solver.
    The ``solve()`` signature is identical to ``RT_SWIKA_FATROP.solve()`` so the two are
    interchangeable behind a simple backend ``if`` switch in the caller.

    Diagnostics:
      * ``solve()`` records, per call, wall time and the CPU core it ran on (plus
        that core's clock frequency, when readable from sysfs). This exists to
        distinguish genuine solve-time variability from OS-scheduler core
        migration / contention with other threads in the same process. Call
        ``print_core_performance_breakdown()`` any time to see a summary.
    """

    def __init__(self, pin_model: pin.Model, keys_to_track: List, N: int, dt: float,
                 dict_dof_to_keypoints: Dict = None, with_freeflyer: bool = True,
                 code: str = 'c', build: bool = True,
                 export_dir: str = None, acados_source_dir: str = None,
                 max_iter: int = None) -> None:
        if AcadosOcpSolver is None:
            raise ImportError(
                "The acados MHE backend was selected but 'acados_template' is not "
                "installed. Install acados (https://docs.acados.org) or set "
                "mhe_backend='fatrop'.")
        if N < 2:
            raise ValueError(f"RT_SWIKA_ACADOS requires N >= 2 (got {N})")
        self._pin_model = pin_model
        self._nq = pin_model.nq
        self._nv = pin_model.nv
        self._nx = self._nq + self._nv
        self._nu = self._nv
        self._with_freeflyer = with_freeflyer
        self._N = N            # number of nodes (RT_SWIKA_FATROP convention)
        self._Nh = N - 1       # acados horizon (number of intervals)
        self._dt = float(dt)   # baked into the discrete dynamics at codegen time
        self._keys_to_track = keys_to_track
        self._n_markers = len(keys_to_track)
        self._nmc = 3 * self._n_markers
        self._code = code
        self._max_iter = max_iter  # shared MHE knob (settings.mhe_max_iter): None=acados default 50; caps SQP iterations
        self._dict_dof_to_keypoints = dict_dof_to_keypoints

        # CasADi symbolic model -- FK is baked from THIS (calibrated) model.
        self._cmodel = cpin.Model(self._pin_model)
        self._cdata = self._cmodel.createData()

        if acados_source_dir:
            os.environ["ACADOS_SOURCE_DIR"] = str(acados_source_dir)
        self._export_dir = export_dir or os.path.join(os.getcwd(), "acados_codegen")
        os.makedirs(self._export_dir, exist_ok=True)
        self._json_path = os.path.join(self._export_dir, f"acados_ocp_ik_N{N}.json")

        self._w_cache = None   # last cost_weights, to skip redundant W updates
        self._ocp_solver = self._create_ocp_solver(build=build)

        # --- lightweight per-call diagnostics (core id / clock / wall time) ---
        self._solve_diag = collections.deque(maxlen=5000)

    def _build_marker_fk_expr(self, cq):
        """CasADi expression of stacked marker positions [x0,y0,z0, x1,...] for q."""
        cpin.framesForwardKinematics(self._cmodel, self._cdata, cq)
        n_frames = len(self._pin_model.frames.tolist())
        cols = []
        for key in self._keys_to_track:
            idx = self._cmodel.getFrameId(key)
            if idx < n_frames:  # frame exists in the model
                cols.append(self._cdata.oMf[idx].translation)
        return casadi.vertcat(*cols)

    @staticmethod
    def _build_block_weight(w_markers, w_state, w_control, nmc, nx, nu, terminal=False):
        """Block-diagonal NONLINEAR_LS weight reproducing RT_SWIKA_FATROP's cost."""
        if terminal:
            ny = nmc + nx
            W = np.zeros((ny, ny))
            W[:nmc, :nmc] = w_markers * np.eye(nmc)
            W[nmc:nmc + nx, nmc:nmc + nx] = w_state * np.eye(nx)
            return W
        ny = nmc + nx + nu
        W = np.zeros((ny, ny))
        W[:nmc, :nmc] = w_markers * np.eye(nmc)
        W[nmc:nmc + nx, nmc:nmc + nx] = w_state * np.eye(nx)
        W[nmc + nx:, nmc + nx:] = w_control * np.eye(nu)
        return W

    def _create_ocp_solver(self, build: bool):
        if "ACADOS_SOURCE_DIR" not in os.environ:
            raise EnvironmentError(
                "ACADOS_SOURCE_DIR is not set. Point it at your acados install "
                "(export ACADOS_SOURCE_DIR=/path/to/acados, or pass "
                "acados_source_dir=...) before constructing RT_SWIKA_ACADOS.")

        nmc, nx, nu = self._nmc, self._nx, self._nu

        # ── CasADi symbolics ──
        cx = casadi.SX.sym("x", nx)
        cu = casadi.SX.sym("u", nu)
        cq = cx[:self._nq]
        cdq = cx[self._nq:]

        # Discrete Euler dynamics (identical to RT_SWIKA_FATROP)
        q_next = cpin.integrate(self._cmodel, cq, cdq * self._dt)
        dq_next = cdq + cu * self._dt
        x_next = casadi.vertcat(q_next, dq_next)

        markers_expr = self._build_marker_fk_expr(cq)

        # ── Acados model ──
        model = AcadosModel()
        model.name = f"rt_swika_acados_nq{self._nq}_N{self._N}"
        model.x = cx
        model.u = cu
        model.disc_dyn_expr = x_next
        # NONLINEAR_LS residuals: stage [markers(q), x, u], terminal [markers(q), x]
        model.cost_y_expr = casadi.vertcat(markers_expr, cx, cu)
        model.cost_y_expr_e = casadi.vertcat(markers_expr, cx)

        # ── Acados OCP ──
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self._Nh
        ocp.solver_options.tf = self._Nh * self._dt

        # Cost placeholders; real values set per-solve from cost_weights.
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = self._build_block_weight(1.0, 1e-3, 1e-5, nmc, nx, nu)
        ocp.cost.W_e = self._build_block_weight(1.0, 1e-3, 1e-5, nmc, nx, nu, terminal=True)
        ocp.cost.yref = np.zeros(nmc + nx + nu)
        ocp.cost.yref_e = np.zeros(nmc + nx)

        # Joint limits on q (freeflyer 0..6 skipped), at stage and terminal nodes.
        if self._with_freeflyer:
            q_con = cx[7:self._nq]
            lo = np.array(self._pin_model.lowerPositionLimit[7:self._nq])
            hi = np.array(self._pin_model.upperPositionLimit[7:self._nq])
        else:
            q_con = cx[:self._nq]
            lo = np.array(self._pin_model.lowerPositionLimit[:self._nq])
            hi = np.array(self._pin_model.upperPositionLimit[:self._nq])
        if q_con.shape[0] > 0:
            model.con_h_expr = q_con
            ocp.constraints.lh = lo
            ocp.constraints.uh = hi
            model.con_h_expr_e = q_con
            ocp.constraints.lh_e = lo
            ocp.constraints.uh_e = hi

        # NOTE: ocp.constraints.x0 is intentionally NOT set -> the arrival cost
        # stays soft (w1 ||x_0 - X0||^2), matching RT_SWIKA_FATROP (no hard clamp).

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = self._max_iter if self._max_iter is not None else 50
        ocp.solver_options.qp_solver_iter_max = 10
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"

        ocp.code_export_directory = self._export_dir
        return AcadosOcpSolver(ocp, json_file=self._json_path,
                               generate=build, build=build)

    def solve(self, X: np.ndarray, U: np.ndarray, marker_meas: np.ndarray,
              X0: np.ndarray, cost_weights, dt: float):
        """Drop-in replacement for ``RT_SWIKA_FATROP.solve`` (identical I/O)."""
        _core_before = _get_current_cpu_core()
        _wall_t0 = time.perf_counter()

        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        marker_meas = np.asarray(marker_meas, dtype=float)
        X0 = np.asarray(X0, dtype=float).flatten()

        if abs(float(dt) - self._dt) > 1e-9:
            raise ValueError(
                f"RT_SWIKA_ACADOS was code-generated with dt={self._dt} but solve() "
                f"received dt={dt}. dt is baked into the discrete dynamics; rebuild "
                f"the solver to use a different dt.")

        w = np.asarray(cost_weights, dtype=float).flatten()
        w0, w1, w2 = w[0], w[1], w[2]

        # Update cost weights only when they change (constant in the pipeline).
        if self._w_cache is None or not np.array_equal(w, self._w_cache):
            W = self._build_block_weight(w0, w1, w2, self._nmc, self._nx, self._nu)
            W_e = self._build_block_weight(w0, w1, w2, self._nmc, self._nx, self._nu,
                                           terminal=True)
            for k in range(self._Nh):
                self._ocp_solver.cost_set(k, "W", W)
            self._ocp_solver.cost_set(self._Nh, "W", W_e)
            self._w_cache = w.copy()

        # Warm start from the incoming trajectory (pipeline carries previous solution).
        for k in range(self._N):
            self._ocp_solver.set(k, "x", np.ascontiguousarray(X[:, k]))
        for k in range(self._Nh):
            self._ocp_solver.set(k, "u", np.ascontiguousarray(U[:, k]))

        # References: per-node marker target + soft state anchor X0 (+ zero ctrl ref).
        zeros_u = np.zeros(self._nu)
        for k in range(self._Nh):
            yref_k = np.concatenate([marker_meas[:, k], X0, zeros_u])
            self._ocp_solver.cost_set(k, "yref", yref_k)
        yref_e = np.concatenate([marker_meas[:, self._N - 1], X0])
        self._ocp_solver.cost_set(self._Nh, "yref", yref_e)

        _wall_before_c_solve = time.perf_counter()
        self._ocp_solver.solve()  # status 0=success, 2=max_iter (best iterate usable)
        _wall_after_c_solve = time.perf_counter()

        X_out = np.zeros((self._nx, self._N))
        U_out = np.zeros((self._nu, self._N))
        for k in range(self._N):
            X_out[:, k] = self._ocp_solver.get(k, "x")
        for k in range(self._Nh):
            U_out[:, k] = self._ocp_solver.get(k, "u")
        # U_out[:, -1] stays 0: RT_SWIKA_FATROP's terminal control is an unused free var.

        _wall_t1 = time.perf_counter()
        _core_after = _get_current_cpu_core()
        _freq_khz = _read_cpu_freq_khz(_core_after)

        self._solve_diag.append({
            "wall_ms": (_wall_t1 - _wall_t0) * 1000.0,
            "c_solve_ms": (_wall_after_c_solve - _wall_before_c_solve) * 1000.0,
            "core_id": _core_after,
            "core_migrated": _core_before != _core_after,
            "freq_mhz": (_freq_khz / 1000.0) if _freq_khz is not None else None,
        })

        return X_out, U_out

    def print_core_performance_breakdown(self) -> None:
        """Per-core breakdown of solve() latency, migrations, and clock speed.

        Use this to tell apart genuine solve-time variability from OS-scheduler
        contention: if migrations are frequent, or latency/clock differ sharply
        between cores, that points at contention (fix: thread/process pinning)
        rather than the solver itself needing more/fewer iterations.
        """
        rows = list(self._solve_diag)
        if not rows:
            print("[RT_SWIKA_ACADOS] no solve() calls recorded yet")
            return

        total = len(rows)
        core_latencies = collections.defaultdict(list)
        core_freqs = collections.defaultdict(list)
        for r in rows:
            core_latencies[r["core_id"]].append(r["wall_ms"])
            if r["freq_mhz"] is not None:
                core_freqs[r["core_id"]].append(r["freq_mhz"])
        migrations = sum(1 for r in rows if r["core_migrated"])

        wall = [r["wall_ms"] for r in rows]
        c_solve = [r["c_solve_ms"] for r in rows]

        print(f"\n{'='*20} RT_SWIKA_ACADOS solve() diagnostics ({total} calls) {'='*20}")
        print(f"total wall (ms):   mean {statistics.mean(wall):8.2f} | median {statistics.median(wall):8.2f} | max {max(wall):8.2f}")
        print(f"of which C solve:  mean {statistics.mean(c_solve):8.2f} | median {statistics.median(c_solve):8.2f} | max {max(c_solve):8.2f}")
        print(f"core migrations: {migrations}/{total} calls switched cores | distinct cores used: {sorted(core_latencies.keys())}")
        print(f"\n{'Core ID':<9} | {'Calls':<7} | {'Share':<8} | {'Median (ms)':<12} | {'Max (ms)':<10} | {'Avg Clock':<10}")
        print("-" * 70)
        for cid in sorted(core_latencies.keys()):
            times = core_latencies[cid]
            freqs = core_freqs[cid]
            share = 100.0 * len(times) / total
            freq_str = f"{statistics.mean(freqs):.0f} MHz" if freqs else "N/A"
            print(f"{cid:<9} | {len(times):<7} | {share:6.1f}% | {statistics.median(times):10.2f}  | {max(times):8.2f}  | {freq_str}")

        if len(core_latencies) > 1:
            med_per_core = {cid: statistics.median(t) for cid, t in core_latencies.items()}
            fastest, slowest = min(med_per_core, key=med_per_core.get), max(med_per_core, key=med_per_core.get)
            delta = med_per_core[slowest] - med_per_core[fastest]
            print(f"\nGap between fastest (core {fastest}) and slowest (core {slowest}) median: {delta:.2f} ms")
            if delta > 15.0 or migrations > total * 0.1:
                print("-> High variance/migration detected: likely OS-scheduler contention.")
                print("   Consider pinning this thread's affinity away from GPU-feeding threads.")