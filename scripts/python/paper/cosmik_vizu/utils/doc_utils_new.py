import casadi
from pinocchio import casadi as cpin
import pinocchio as pin
import numpy as np
from scipy import signal
import pandas as pd
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from pinocchio import log3
 
 
 
 
class DocHumanMotionGeneration_InvDyn:
    
    def __init__(self,model,param):
         
       
        
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        
        self.model = model = pin.Model(model)
        self.data = data = model.createData()
         
        self.param=param
        
        self.nq = cmodel.nq 
        self.nv = cmodel.nv 

        # * Define the problem dimensions
        # We will define a state x = (q, v)^T to describe the robot dynamics
        nx = self.nq + self.nv     # state dimension: positions and velocities
        nu = self.nv#-6         # control dimension: the accelerations

        # * Create casadi symbolic variables
        # These variables are used to define symbolic expression and are replaced in the solver by some values according to the decision variables
        cx = casadi.SX.sym("x", nx, 1) # state: the positions and velocities
        cu = casadi.SX.sym("u", nu, 1) # control: the accelerations
        
        

        M_target = pin.SE3(param.FOI_orientation[0],param.FOI_position[0]) # SE3 position and rotation of the target to reach


        if param.free_flyer:
            
            nf = 12               # Got two 6D forces
            cf = casadi.SX.sym("f", nf, 1) # two 6D external forces at feet level

            # * Definition of multi contact problem
            
            # get feet original pose (supposedly not moving during the entire task)
            M_lf = pin.SE3(param.FOI_orientation[1],param.FOI_position[1]) # SE3 position and rotation of the left foot to hold
            M_rf = pin.SE3(param.FOI_orientation[2],param.FOI_position[2]) # SE3 position and rotation of the right foot to hold

            # get the free flyer and feet symbolic poses
            oMf_ff = cdata.oMi[1]  # placement of root joint in world

            id_lf = model.getFrameId(param.FOI_to_set[1])   # The left foot has to stay in contact with the floor
            id_rf = model.getFrameId(param.FOI_to_set[2])     # The right foot has to stay in contact with the floor


            self.oMf_lf = cdata.oMf[id_lf]  # placement of left foot in world frame
            self.oMf_rf = cdata.oMf[id_rf ]  # placement of right foot in world frame
            
            force_lf = casadi.SX.sym("force_lf", 6)  # 6D spatial force  
            force_rf = casadi.SX.sym("force_rf", 6)  # 6D spatial force  
            
            cframe_lf = cmodel.frames[id_lf] #left foot frame
            cframe_rf = cmodel.frames[id_rf] #rigth foot  frame
            
            # * Build contact forces list
            forces = [ cpin.Force.Zero() for _ in cmodel.joints] #Initializes a list of zero forces, one per joint
            forces[cframe_lf.parentJoint] = cframe_lf.placement.act(cpin.Force(cf[0:6])) # cf[0:6] is the left foot wrench (spatial force: 3 force + 3 torque)
            forces[cframe_rf.parentJoint] = cframe_rf.placement.act(cpin.Force(cf[6:])) 

            cforces = cpin.StdVec_Force()
            for f in forces:
                cforces.append(f)

           
            #### Calculation related to contraints
        
            self.com = casadi.Function('com', [cx, cu], [self.cdata.com[0]]) # CoM
            self.vcom = casadi.Function('vcom', [cx, cu], [self.cdata.vcom[0]]) # CoM velocity
              
             
            if self.param.external_forces=="linear_zmp_distance_forces_estimation":# if the external GRFM under each foot at linearly interpolated
                
                cf_ff=casadi.SX.sym("f_ff", 6)
                
                total_force_ff = cpin.Force(cf_ff)#casadi.SX.sym("f_ff", 6))  # spatial force for free flyer 
                total_force_world=oMf_ff.act(total_force_ff)
            
            
                f = total_force_world.linear
                tau = total_force_world.angular
                
                fz = f[2] + 1e-8  # Add epsilon to avoid division by zero

                x_zmp = -tau[1] / fz  # -τ_y / f_z
                y_zmp =  tau[0] / fz  #  τ_x / f_z

                zmp_world = casadi.vertcat(x_zmp, y_zmp)  # ZMP in world frame (ground plane)

                # Compute distances in world frame and weights 
                dl = casadi.norm_2( zmp_world - self.oMf_lf.translation[0:1])
                dr = casadi.norm_2( zmp_world - self.oMf_rf.translation[0:1])
                denom = dl + dr + 1e-8  # Avoid division by zero
            
                wl = dr / denom
                wr = dl / denom

                # Partition world-frame wrench 
                fl_world = wl * total_force_world #Note: the weight of the left foot depends on the distance to the right and vice versa — this ensures that if the ZMP is closer to the left foot, it bears more load.
                fr_world = wr * total_force_world

                # Express in local foot frames
                fl_local = self.oMf_lf.actInv(fl_world)
                fr_local = self.oMf_rf.actInv(fr_world)

                #  # Create CasADi function
                self.fl_local = casadi.Function("fl_local", [cx, cu, cf_ff], [ fl_local.vector ])
                self.fr_local = casadi.Function("fr_local", [cx, cu, cf_ff], [ fr_local.vector ])
        
            
            
            if self.param.external_forces=="optimal_forces_estimation":# if the external GRFM under each foot are decision variables of the ocp
                # Transform from foot to freeflyer
                lffMf = oMf_ff.inverse() * self.oMf_lf
                rffMf = oMf_ff.inverse() * self.oMf_rf
            
                # Express feet forces in the freeflyer frame
                force_lf_at_ff = lffMf.act(cpin.Force(force_lf))
                force_rf_at_ff = rffMf.act(cpin.Force(force_rf))
                
                # Create CasADi function
                self.lf_force_at_ff = casadi.Function("left_force_at_freeflyer", [cx, cu, force_lf], [force_lf_at_ff.vector])
                self.rf_force_at_ff = casadi.Function("right_force_at_freeflyer", [cx, cu, force_rf], [force_rf_at_ff.vector])

            
                # # CoP a calculation
                # # Extract vertical force (FY) and torques (Mx, MZ) in local frame

                # Fy_l = cf[1]# Y is vertical axis in the local frame
                # Mx_l = cf[3]
                # Mz_l = cf[5]

                # Fy_r = cf[1+6]
                # Mx_r = cf[3+6]
                # Mz_r = cf[5+6]

                # cop_l_local = casadi.vertcat(Mz_l / Fy_l, -Mx_l / Fy_l, 0)
                # cop_r_local = casadi.vertcat(Mz_r / Fy_r, -Mx_r / Fy_r, 0)

                # cop_l_global=casadi.mtimes(self.oMf_lf.rotation, cop_l_local)+ self.oMf_lf.translation
                # cop_r_global=casadi.mtimes(self.oMf_rf.rotation, cop_r_local)+ self.oMf_rf.translation

                # F_lg=self.oMf_lf.rotation@cf[0:3] #left forces in the GSR
                # F_rg=self.oMf_rf.rotation@cf[6:9] #right  forces in the GSR
            
                # self.F_lg=casadi.Function("F_lg", [cx,cu,cf], [F_lg])
                # self.F_rg=casadi.Function("F_rg", [cx,cu,cf], [F_rg])
            
                # total_cop = ( F_lg[2]* cop_l_global +  F_rg[2] * cop_r_global) / (F_lg[2] + F_rg[2] + 1e-6) # barycenter to get the total cop
                # total_cop[2] =0
                # self.total_cop=casadi.Function("total_cop", [cx,cu,cf], [total_cop])
        
        # * Contraint on the contacts
            # Position error
            self.pos_error = casadi.Function('placement_contact_error', [cx, cu], [self.approx_log6(id_lf, M_lf,cx,cu)])
            self.pos_error2 = casadi.Function('placement_contact_error2', [cx, cu], [self.approx_log6(id_rf, M_rf,cx,cu)])
            # Velocity error
            self.vel_error = casadi.Function('velocity_contact_error', [cx, cu], [cpin.getFrameVelocity(cmodel, cdata, id_lf, pin.LOCAL).vector]) # Target velocity is null in the world frame so difference is the effector frame velocity
            self.vel_error2 = casadi.Function('velocity_contact_error2', [cx, cu], [cpin.getFrameVelocity(cmodel, cdata, id_rf, pin.LOCAL).vector]) # Target velocity is null in the world frame so difference is the effector frame velocity
        
            # Acceleration error
            self.acc_cstr = casadi.Function('acc_constraint_error', [cx, cu], [cpin.getFrameClassicalAcceleration(cmodel, cdata, id_lf, pin.LOCAL).vector])
            self.acc_cstr2 = casadi.Function('acc_constraint_error2', [cx, cu], [cpin.getFrameClassicalAcceleration(cmodel, cdata, id_rf, pin.LOCAL).vector])

        
        
        
        #a = cpin.aba(cmodel, cdata, cx[:self.nq], cx[self.nq:], casadi.vertcat(np.zeros(6),cu), cforces)           # a is a symbolic expression corresponding to the joints acceleration and depending on the symbolic variables cx and cu
        #a = cpin.aba(cmodel, cdata, cx[:self.nq], cx[self.nq:], cu, cforces)           # a is a symbolic expression corresponding to the joints acceleration and depending on the symbolic variables cx and cu
        #cpin.forwardKinematics(cmodel, cdata, cx[:self.nq], cx[self.nq:], a)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a

        cpin.forwardKinematics(cmodel, cdata, cx[:self.nq], cx[self.nq:])  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
        cpin.updateFramePlacements(cmodel, cdata)                   # Update the frames placement to symbolic expressions in data
        
        # Calculations related to the cost functions
        
        self.tau_free=casadi.Function('tau_freeflyer',[cx, cu],[cpin.rnea(cmodel,cdata,cx[:self.nq],cx[self.nq:],cu)  ])# joint torques without external wrench 
        
        if param.free_flyer:
            self.tau=casadi.Function('tau',[cx, cu, cf],[cpin.rnea(cmodel,cdata,cx[:self.nq],cx[self.nq:],cu, cforces)  ])#  joint torques with external wrench 
        else:
            self.tau=casadi.Function('tau',[cx, cu],[cpin.rnea(cmodel,cdata,cx[:self.nq],cx[self.nq:],cu)  ])#  joint torques without external wrench 

        energy=[]
        for j in range(6,self.nv):
            energy+=casadi.fabs(cx[self.nq+j]*self.tau(cx,cu,cf)[j])

        self.energy=casadi.Function('energy',[cx, cu, cf],[ energy  ]) 
        
        dtau_dq, dtau_dv, dtau_da=cpin.computeRNEADerivatives(cmodel,cdata,cx[:self.nq],cx[self.nq:],cu)
        self.dtau=casadi.Function('dtau',[cx, cu, cf],[ dtau_dq@cx[self.nq:]    ])
        # Casadi Functions for cost function definition
        # self.geodesic=casadi.Function('geodesic',[q,dq,tau],[ dq.T@cdata.M@dq   ])
        # self.tip = casadi.Function('tip', [q], [ self.cdata.oMf[-1].translation[[0,2]] ])
        # self.vtip =casadi.Function('vtip', [q,dq,tau], [  cpin.getFrameVelocity(self.cmodel,self.cdata,cmodel.getFrameId('hand') ,cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[[0,2]] ] )
    

        
        


        
        
   
        
        # * Cost on the distance to the target(s)-eventually

        cost_to_target=[]
        cost_to_target.append(self.approx_log6(param.FOI_to_set_Id[0], M_target,cx,cu))
       # cost_to_target.append(self.approx_log6(param.FOI_to_set_Id[1], Mtarget,cx,cu))
       
        self.cost_to_target = casadi.Function('cost_to_target', [cx, cu], [casadi.vertcat( *cost_to_target) ])

        # * Define a function to get the next state from the robot dynamics 
        self.cnext = casadi.Function('cnext', [cx, cu], [cpin.integrate(cmodel, cx[:self.nq], cx[self.nq:]*self.param.dt)])
        qnext=self.cnext(cx,cu)
        dqnext=cx[self.nq:]+cu* self.param.dt
        self.dyn_fun = casadi.Function('dyn', [cx, cu], [casadi.vertcat(qnext,dqnext)])   
             

    
    def solve_doc(self,param):
          
        T=int(param.nb_samples)
         
         # * Define the problem dimensions
        # We will define a state x = (q, v)^T to describe the robot dynamics
        nx = self.nq + self.nv     # state dimension
        ndx = 2*self.nv      # state derivative
        nu = self.nv#-6         # control
        nf = 12                # Got two 6D forces
        
        # * Defining the casadi optimisation problem
        opti = casadi.Opti()

        # * Define decision variables
        x0 = np.concatenate((param.qdi, np.zeros(self.nv)))
        # State
        
       # dxs=[]
        xs=[]
        us=[]
        fs=[]
        for k in range(T):
            xs.append(opti.variable(nx))#self.integrate(x0, dxs[-1]))
            us.append(opti.variable(nu))
            
            if param.free_flyer:
                if self.param.external_forces=="optimal_forces_estimation":
                    fs.append(opti.variable(nf))
                    
                if self.param.external_forces=="linear_zmp_distance_forces_estimation":
                    fs.append(casadi.SX(nf))
                    
        xs.append(opti.variable(nx))
        

        self.xs=xs
        self.fs=fs
        self.us=us

        # * Define the OCP
        total_cost = 0 # Initialisation
        
        opti.subject_to(xs[0][0:self.nq] == param.qdi) 
        opti.subject_to(xs[0][self.nq:] == np.zeros(self.nv))
        opti.subject_to(us[0]  == np.zeros(self.nv))
        epsilon=1e-2
        # opti.subject_to(us[0]  <= epsilon)
        # opti.subject_to(us[0]  >= -epsilon)
        
        # opti.subject_to(xs[0][self.nq:] <= epsilon)
        # opti.subject_to(xs[0][self.nq:] >= -epsilon)
    
        # * Defining the costs for goal driven OCP
        w = [5e3*(t+1)/T for t in range(T)]
        # w = [0 for t in range(T)]; 
        # w[-1] = 5e3
        
       
        for t in range(T): 
             
            # Run for each time step
                     
            x_next = self.dyn_fun(xs[t], us[t])   # Compute the dynamic to get the next state
            #if t!=T-1:
            opti.subject_to( xs[t + 1]  == x_next)
            #opti.subject_to(us[t][0:6] == 0)                           # No control of the free flyer
            
            
            #opti.subject_to(self.acc_cstr(xs[t], us[t], fs[t]) <= epsilon)
            #opti.subject_to(self.acc_cstr(xs[t], us[t], fs[t]) >= -epsilon)
            if t>0 and t<T-2:
                
                for j in range(7,self.nq):
                    opti.subject_to(opti.bounded(param.q_min[j-7], xs[t][j], param.q_max[j-7]))  

                if param.free_flyer:
                    # We add a corrector called the Baumgarte Corrector to handle the drift introduced from numerical integration
                    #opti.subject_to(self.acc_cstr(xs[t], us[t], fs[t]) == -1000*self.pos_error(xs[t], us[t], fs[t]) - 2*np.sqrt(1000)*self.vel_error(xs[t], us[t], fs[t])) # Constraint the relative acceleration to be null
                    #opti.subject_to(self.acc_cstr2(xs[t], us[t], fs[t]) == -1000*self.pos_error2(xs[t], us[t], fs[t]) - 2*np.sqrt(1000)*self.vel_error2(xs[t], us[t], fs[t])) # Constraint the relative acceleration to be null

                    opti.subject_to( 1*(self.acc_cstr(xs[t], us[t] ))      == 0)
                    opti.subject_to(  1*(self.acc_cstr2(xs[t], us[t] ))     == 0)

                    # projected COM constraint
                    #opti.subject_to(opti.bounded(param["FOI_position"][1][0]-0.05, self.com(xs[t], us[t], fs[t])[0], param["FOI_position"][1][0]+0.22)) 

                    # in case use the linear relationship between feet position and cop to estimate external forces under each contact point
                    
                    
                    
                    #opti.subject_to(opti.bounded(param.FOI_position[1][0]-0.05, self.total_cop(xs[t], us[t], fs[t])[0], param.FOI_position[1][0]+0.22)) # foot boundaries 5cm behind the ankle and 22cm in front

                
           
            if param.free_flyer: 
                if self.param.external_forces=="optimal_forces_estimation":#
                    opti.subject_to( self.lf_force_at_ff(xs[t],us[t],fs[t][:6]) +  self.rf_force_at_ff(xs[t],us[t],fs[t][6:])  == self.tau_free(xs[t],us[t])[0:6]) 
                #opti.subject_to( self.F_lg(xs[t], us[t], fs[t])[2]>0)
                #opti.subject_to( self.F_rg(xs[t], us[t], fs[t])[2]>0)
            
                #opti.subject_to( fs[t][1]>0)# vertical left foot force (Y-axis)
                #opti.subject_to( fs[t][1+6]>0)# vertical right foot force
            
           # total_cost +=  +w[t]*casadi.sumsqr(self.cost_to_target(xs[t], us[t], fs[t])[2]) + 1e-5 *(us[t]-self.u0 ).T@(us[t]-self.u0 ) # + 1e-8*fs[t].T@fs[t]  + 1e-6*dxs[t].T@dxs[t]  # Running cost: Weighted sum of the distance to the target and a control cost
            #total_cost += 1e3 *(self.cost_to_target(xs[-1], us[-1], fs[-1])[2]).T@(self.cost_to_target(xs[-1], us[-1], fs[-1])[2] )+  1e3 *(xs[-1][self.nq:]).T@(xs[-1][self.nq:] )  +  1e3 *(us[-1]).T@(us[-1] )  +1e-3 *(self.tau(xs[t], us[t], fs[t])[6:]  ).T@(self.tau(xs[t], us[t], fs[t])[6:]   ) +  1e-5 *(us[t]).T@(us[t] ) +1e-4*fs[t].T@fs[t] 
                if self.param.external_forces=="linear_zmp_distance_forces_estimation":#
                    
                     
                    fl=self.fl_local(xs[t],us[t], self.tau_free(xs[t],us[t])[0:6])
                    fr=self.fr_local(xs[t],us[t], self.tau_free(xs[t],us[t])[0:6])
        
                    fs[t]=casadi.vertcat(fl, fr)
                    
                    
            J=self.calc(xs[t],us[t],fs[t])
            
            cost_idx=0
            for cost in  self.param.active_costs:
                if cost=="min_joint_torque":
                    total_cost +=param.weights[cost] *( J[cost_idx]   )
                    cost_idx+=1
                    
                if cost=="min_joint_vel":
                    print("min vel")
                    cost_idx+=1
                if cost=="min_joint_acc":
                    total_cost +=param.weights[cost] *( J[cost_idx] )
                    cost_idx+=1
                     
            
            if t==param.FOI_sample[0]:#Impose  position and zero velocity contraints at the desired sample of time
                opti.subject_to( (self.cost_to_target(xs[param.FOI_sample[0]], us[param.FOI_sample[0]])[2])   == 0) # hard constraint on the vertical position of the FOI (pelvis)
                opti.subject_to(xs[param.FOI_sample[0]][self.nq:] == np.zeros(self.nv)) # last velocity is set to 0
                 
            # Costs related to constraints 
            total_cost += (w[t] *(self.cost_to_target(xs[param.FOI_sample[0]], us[param.FOI_sample[0]])[2]).T@(self.cost_to_target(xs[param.FOI_sample[0]], us[param.FOI_sample[0]])[2] )+
            1e-4*fs[t].T@fs[t] )# GRFM regularization term
         
        
        #opti.subject_to(xs[-1][0:self.nq] == param.qdi)# go back to standing position
       # opti.subject_to(us[-1] == np.zeros(self.nv)) # last acc is set to 0
        # opti.subject_to(us[-1]  <= epsilon)
        # opti.subject_to(us[-1]  >= -epsilon)
        
        # opti.subject_to(xs[-1][self.nq:] <= epsilon)
        # opti.subject_to(xs[-1][self.nq:] >= -epsilon)
            
        #opti.subject_to( self.cost_to_target(xs[-1], us[-1], fs[-1])[2] == 0)
        #opti.subject_to(xs[-1][self.nq:] == np.zeros(self.nv)) # last velocity is set to 0
        
        # * Solve the problem
        opti.minimize(total_cost)

        # set initial conditions
        
    
        for x in xs:
       
            x0 = np.concatenate([param.qdi, np.zeros(self.nv)])
            opti.set_initial(x, x0)# uprigth standing
        
        if self.param.external_forces=="optimal_forces_estimation":#
                  
            # divide the external wrench at initial joint configuration in two
            tau0_ff=self.tau_free(x0,np.zeros(self.nv))[0:6]
            pin.forwardKinematics(self.model, self.data, param.qdi)
            pin.updateFramePlacements(self.model, self.data)

            # Frame indices of the feet
            id_lf = self.model.getFrameId("left_foot")
            id_rf = self.model.getFrameId("right_foot")

            # SE(3) transforms
            Mb_lf = self.data.oMf[id_lf].actInv(self.data.oMf[self.model.getFrameId("root_joint")])  # base->left foot
            Mb_rf = self.data.oMf[id_rf].actInv(self.data.oMf[self.model.getFrameId("root_joint")])  # base->right foot
            
            tau_0LF=self.transport_effort(np.asarray(Mb_lf.rotation),np.asarray(Mb_lf.translation),np.asarray(tau0_ff/2))
            tau_0RF=self.transport_effort(np.asarray(Mb_rf.rotation),np.asarray(Mb_rf.translation.T),np.asarray(tau0_ff/2))

            for f in fs:     
                opti.set_initial(f, np.concatenate([tau_0LF,tau_0RF]) )# weigth in standing posture
            
            
        for u in us: opti.set_initial(u, np.zeros(self.nv))#self.u0 ) # Set initial guess for the acc control
       
        if self.param.solver=="ipopt":
            #Solver options
            opts = {
                    "ipopt.print_level": 5,  # Suppress solver output
                    "ipopt.sb": "yes",  # Suppress banner
                    "ipopt.max_iter": 1000,  # Maximum iterations
                    "ipopt.linear_solver": "mumps",  # Linear solver
                    "print_time": 2,  # Print timing information
                    "expand": True,  # Expand expressions for better performance
                    "ipopt.hessian_approximation": "limited-memory",  # Hessian approximation
                    "ipopt.tol": 1e-3,  # Overall tolerance
                    "ipopt.constr_viol_tol": 1e-3,  # Constraint violation tolerance
                    "ipopt.compl_inf_tol": 1e-3,  # Complementarity tolerance
                    "ipopt.dual_inf_tol": 1e-3,  # Dual infeasibility tolerance
                    "ipopt.acceptable_tol": 1e-3,  # Acceptable tolerance
                    "ipopt.acceptable_constr_viol_tol": 1e-3  # Acceptable constraint violation tolerance
                }
            # ,"jit": False, "jit_options": jit_options}#{'ipopt.print_level': 0}#, 'linear_solver':'mumps'}#"expand":True, 'ipopt.hessian_approximation':'limited-memory'}
                
            opti.solver("ipopt",opts)   
        
      
        if self.param.solver=="fatrop":

            ### Define the solver
            options = {}
            options["verbose_init"] = False
            options["verbose"] = False
            options["print_time"] = False
            options["expand"] = True
            # options["fatrop"]"fatrop.hessian_approximation": "limited-memory"
            #options["fatrop"] = {"print_level":5, "max_iter":1000, "mu_init": 1e-5, 'warm_start_mult_bound_push' : 1e-7, "bound_push":1e-7, "tol":1e-3, }#, "linsol_iterative_refinement":False}#, "warm_start_init_point":True}
            options["structure_detection"] = "auto"
            options["debug"] = False

            opti.solver("fatrop", options)
            # plt.ion()  # Activate interactive plotting
            # # # Add the callback
            # self.opti=opti
            # opti.callback(self.callback)    
        
        
        
        sol = opti.solve_limited()
        
        
        
        print("Solution found you can now visualize it in the viewer")
        xs_sol = np.array([opti.value(x) for x in xs])  # Get the optimal values of the decision variables
        us_sol = np.array([opti.value(u) for u in us])
        fs_sol = np.array([opti.value(f) for f in fs])  
        
        return xs_sol, us_sol, fs_sol
    
    def tau_free_eval(self,xs,cu,param):
        tau=self.tau_free(xs,cu)
        return tau 
    
    def transport_effort(self,R,p,tau_A):
        #transport 6D effort from point A to frame B defined by its R,p pose
        p=p.reshape(3,)
       
        F=(R  @ tau_A[:3]).reshape(3,)
        M=(R @ tau_A[3:6]).reshape(3,)+ np.cross(p, F)
        tau_B=np.concatenate((F,M))
        return tau_B



    def calc(self,cx,cu,cf):
            ### WARNING REMEMBER TO HANDLE TEH NO FREEFLEYER CASE !!!!
        self.J=[]
        # if self.param.individual_joint_torques_cost==1:
        #     for j in range(6,self.nv):
        #         self.J.append(self.tau(cx,cu,cf)[j].T@self.tau(cx,cu,cf)[j]) # min torque #C1
        # else:
        for cost in  self.param.active_costs:
            if cost=="min_joint_torque":
                if self.param.groups_joint_torques["all"]==True:
                    self.J.append(self.tau(cx,cu,cf)[6:].T@self.tau(cx,cu,cf)[6:]) # min torque #C1
                else:
                    for group_name, joint_list in self.param.groups_joint_torques.items():
                        if group_name == "all":
                            continue  # skip the 'all' key
                        for joint_name in joint_list:
                            joint_id = self.model.getJointId(joint_name)
                            self.J.append(self.tau(cx,cu,cf)[joint_id].T@self.tau(cx,cu,cf)[joint_id])
                            
            if cost=="min_joint_vel":
                self.J.append(cx[self.nv+6:].T@cx[self.nv+6:]) # min joint velocity, ie NO freeflyer #C2
        
            
            if cost=="min_joint_acc":
                self.J.append(cu[self.nv+6:].T@cu[self.nv+6:]) # min joint accleration, ie NO freeflyer #C3
        #self.J.append( (self.vtip(q,dq,ddq)[0].T@self.vtip(q,dq,ddq)[0]+self.vtip(q,dq,ddq)[1].T@self.vtip(q,dq,ddq)[1])/10 ) # min cartesian vel #C4
        #self.J.append( (self.dtau(cx,cu,cf).T@self.dtau(cx,cu,cf)) ) # min torque change #C5
        #self.J.append((self.energy(cx,cu,cf)) ) # min energy #C6
        #self.J.append((self.geodesic(q,dq,ddq))/4) # min geodesic #C7
        
        return self.J 



    def callback(self,i):
        # Optimization callback function to display while debug
        
        # Evaluate Opti variables using current debug values
        #f_vals=self.opti.debug.value(self.fs)
        x_vals = [self.opti.debug.value(x) for x in self.xs]
        u_vals = [self.opti.debug.value(u) for u in self.us]
        f_vals = [self.opti.debug.value(f) for f in self.fs]
        q_vals = np.column_stack([x[:self.nq] for x in x_vals])  # shape (nq, N)
        
        acc_cte=[]
        for t in range(30):
            
            acc_cte.append(self.acc_cstr(x_vals[t], u_vals[t], f_vals[t]))
            
        acc_cte=np.squeeze(acc_cte)
        
        plt.clf()
         
        #plt.title(f"Iteration {self.iter} - XS (states)")
        plt.plot(acc_cte, 'r-', lw=1.0, label='q')
        plt.grid()
        
        
    #     N = q_vals.shape[1]

    # # build the flat limits (repeat each limit N times)
    #     flat_min = np.repeat(np.concatenate([np.ones(7),self.q_min]), N)
    #     flat_max = np.repeat(np.concatenate([-np.ones(7),self.q_max]), N)
    #     flat_q = q_vals.flatten(order='C')
    #     print("CALLBACK !!!!")


    #     #plt.figure(1)
    #     plt.clf()
         
    #     #plt.title(f"Iteration {self.iter} - XS (states)")
    #     plt.plot(flat_q, 'r-', lw=1.0, label='q')
    #     plt.plot(flat_min, 'k--', lw=0.8, label='q_min/q_max')
    #     plt.plot(flat_max, 'k--', lw=0.8)
    #     plt.grid()
        #plt.legend(["FX","FY","FZ"])
        # Slight pause to update plot
 
        #plt.pause(0.5)
        input("Press Enter to continue...")  # Pause until Enter is pressed
        # tau_0=[]
        # for t in range(20):
        #     tau_0.append( self.tau_free(x_vals[t],u_vals[t],0)[0:6] )
            
        # tau_0=np.array(tau_0)
        # tau_0=np.squeeze(tau_0)
         
  


        # plt.figure(1)
        # plt.clf()
        # plt.subplot(2, 1, 1)
        # #plt.title(f"Iteration {self.iter} - XS (states)")
        # plt.plot(tau_0[:,0:3] )
        # plt.grid()
        # plt.legend(["FX","FY","FZ"])
        #  # Slight pause to update plot

        # plt.subplot(2, 1, 2)
        # plt.plot(tau_0[:,3:] )
        # plt.grid()
        # plt.legend(["MX","MY","MZ"])
        # plt.pause(0.5)
        
        # plt.pause(0.05)
        return casadi.DM(0)
    
  
    
    
    def symbolic_log3(self,R):
        """Symbolic log3 for a CasADi rotation matrix."""
        cos_theta = (casadi.trace(R) - 1) / 2
        cos_theta = casadi.fmin(casadi.fmax(cos_theta, -1), 1)  # clamp to [-1, 1]
        theta = casadi.acos(cos_theta)

        # To avoid division by zero:
        if isinstance(theta, casadi.SX) or isinstance(theta, casadi.MX):
            near_zero = casadi.logic_and(theta < 1e-6, theta > -1e-6)
        else:
            near_zero = abs(theta) < 1e-6

        # Skew-symmetric part
        omega_hat = (R - R.T) / (2 * casadi.sin(theta))

        # Extract vector from skew-symmetric matrix
        omega = casadi.vertcat(omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0])

        # Handle small angle limit
        omega = casadi.if_else(near_zero, casadi.SX.zeros(3), theta * omega)

        return omega
    
    def approx_log6(self,id, M,cx,cu):
        # Update data symbolically inside the function
        cpin.forwardKinematics(self.cmodel, self.cdata, cx[:self.nq], cx[self.nq:],cu)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
        cpin.updateFramePlacements(self.cmodel, self.cdata)                   # Update the frames placement to symbolic expressions in data
       
        tran_error = self.cdata.oMf[id].translation - M.translation
        #rot_error = casadi.diag(self.cdata.oMf[id].rotation.T @ M.rotation)-casadi.SX.ones(3)
        
        R_err = self.cdata.oMf[id].rotation.T @ M.rotation
        rot_error = self.symbolic_log3(R_err)  # returns a 3D vector in so(3)
        
        return(casadi.vertcat(tran_error, rot_error))      
    
  
         
    def solve_initial_control_condition(self, param):
        # model.gravity = pin.Motion(np.zeros(6))  # ! Disable gravity
        # * Get the controls allowing the robot to stay static under gravity
        cu = casadi.SX.sym("u", self.nv, 1)
        acc = casadi.Function("acc", [cu], [cpin.aba(self.cmodel, self.cdata, casadi.SX(param["q0"][0]), casadi.SX.zeros(self.nv), cu)])
        opti = casadi.Opti()                    # Create optimisation object
        us = opti.variable(self.nv)         # Control as a decision variable
        cost = casadi.sumsqr(acc(us))           # Cost - Acceleration of the joints
        opti.subject_to(us[:6]==0)              # Do not control the free flyer
        opti.minimize(cost)                     
        opti.solver("ipopt")                    # Define the solver to use - IPOPT
        sol = opti.solve_limited()              # Run the optimisation process
        u0 = opti.value(us)                     # Get the optimal value found  
        
        return u0
     
 
 

    
    
    
    
    
    
    
    
    
    
    
    