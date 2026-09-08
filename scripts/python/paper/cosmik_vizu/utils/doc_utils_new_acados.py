import casadi
from pinocchio import casadi as cpin
import pinocchio as pin
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from matplotlib import cm
from acados_template import AcadosOcp, AcadosOcpSolver

from pprint import pprint

 
 
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
        nu = self.nv               # control dimension: the accelerations including the free-flyer
        

        # * Create casadi symbolic variables
        # These variables are used to define symbolic expression and are replaced in the solver by some values according to the decision variables
        cx = casadi.SX.sym("x", nx, 1) # state: the positions and velocities
        cu = casadi.SX.sym("u", nu, 1) # control: the accelerations

        # SE3 position and rotation of the n targets to reach
        M_target = [pin.SE3(param.FOI_orientation[i], param.FOI_position[i]).copy()
        for i in range(len(param.FOI_orientation)) ]

        #### Calculation related to contraints
        cpin.forwardKinematics(self.cmodel, self.cdata, cx[:self.nq], cx[self.nq:],cu)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
        cpin.updateFramePlacements(self.cmodel, self.cdata)                   # Update the frames placement to symbolic expressions in data
        cpin.centerOfMass(self.cmodel, self.cdata, cx[:self.nq]) 
            
        self.com = casadi.Function('com', [cx,cu], [self.cdata.com[0]]) # CoM
        self.vcom = casadi.Function('vcom', [cx,cu], [self.cdata.vcom[0]]) # CoM velocity
            
        
        if param.free_flyer:
            # * Definition of multi contact problem
            
            nf = 12               # Got two 6D forces
            cf = casadi.SX.sym("f", nf, 1) # two 6D external forces at feet level
            
            
            id_lf = model.getFrameId("left_foot")   # The left foot has to stay in contact with the floor
            id_rf = model.getFrameId("right_foot")     # The right foot has to stay in contact with the floor
            #q0=pin.neutral(model) 
            # get feet original pose (supposedly not moving during the entire task)
            pin.forwardKinematics(model, data, param.qdi)
            pin.updateFramePlacements(model, data)
            #self.M_lf = data.oMf[id_lf]   # placement of left foot in world frame
            #self.M_rf = data.oMf[id_rf]  # placement of right foot in world frame
            
            self.M_lf = pin.SE3(param.lfi_orientation,param.lfi_position) # SE3 position and rotation of the left foot to hold
            self.M_rf = pin.SE3(param.rfi_orientation,param.rfi_position) # SE3 position and rotation of the right foot to hold

            # get COP base of support limits
            self.BoS_min=np.zeros(2)
            for ax in range(2):# XY axes
                if self.M_lf.translation[ax]<=self.M_rf.translation[ax]:
                    self.BoS_min[ax]=self.M_lf.translation[ax]
                else:
                    self.BoS_min[ax]=self.M_rf.translation[ax]
                    
            self.BoS_max=np.zeros(2)
            for ax in range(2):# XY axes
                if self.M_lf.translation[ax]>=self.M_rf.translation[ax]:
                    self.BoS_max[ax]=self.M_lf.translation[ax]
                else:
                    self.BoS_max[ax]=self.M_rf.translation[ax]
                 
            # get the free flyer and feet symbolic poses
            oMf_ff = cdata.oMi[1]  # placement of root joint in world

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
            self.pos_error_lf = casadi.Function('placement_contact_error_lf', [cx, cu], [self.approx_log6(id_lf, self.M_lf,cx,cu)])
            self.pos_error_rf = casadi.Function('placement_contact_error_rf', [cx, cu], [self.approx_log6(id_rf, self.M_rf,cx,cu)])
            
            # Velocity error
            self.vel_error_lf = casadi.Function('velocity_contact_error_lf', [cx, cu], [cpin.getFrameVelocity(cmodel, cdata, id_lf, pin.LOCAL).vector]) # Target velocity is null in the world frame so difference is the effector frame velocity
            self.vel_error_rf = casadi.Function('velocity_contact_error_rf', [cx, cu], [cpin.getFrameVelocity(cmodel, cdata, id_rf, pin.LOCAL).vector]) # Target velocity is null in the world frame so difference is the effector frame velocity
        
            # Acceleration error
            self.acc_cstr_lf = casadi.Function('acc_constraint_error_lf', [cx, cu], [cpin.getFrameClassicalAcceleration(cmodel, cdata, id_lf, pin.LOCAL).vector])
            self.acc_cstr_rf = casadi.Function('acc_constraint_error_rf', [cx, cu], [cpin.getFrameClassicalAcceleration(cmodel, cdata, id_rf, pin.LOCAL).vector])

        # to be used for torque control
        # a = cpin.aba(cmodel, cdata, cx[:self.nq], cx[self.nq:], cu, cforces)           # a is a symbolic expression corresponding to the joints acceleration and depending on the symbolic variables cx and cu
        # cpin.forwardKinematics(cmodel, cdata, cx[:self.nq], cx[self.nq:], a)  # Update the cdata values with symbolic expressions depending on the robot kinematics and on a
        # cpin.updateFramePlacements(cmodel, cdata)                   # Update the frames placement to symbolic expressions in data
        # self.acc = casadi.Function( "xdot", [cx, cu, cf], [a])  # Casadi function: Takes values of cx and cu and returns corresponding value of the symbolic expression a



        
        
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
        # Loop over all pairs (FOI id, target)
        cost_to_target=[]
        for foi_id, target in zip(param.FOI_to_set_Id, M_target):
            cost_to_target.append(self.approx_log6(foi_id, target, cx, cu))
        
        self.cost_to_target = casadi.Function('cost_to_target', [cx, cu], [casadi.vertcat( *cost_to_target) ])

        
        def normalize_quat(q):
            quat = q[3:7]
            quat_norm = quat / casadi.norm_2(quat)
            return casadi.vertcat(q[:3], quat_norm, q[7:])
        
    
        # Normalize quaternion part of q before integration
        #q_in = normalize_quat(cx[:self.nq])
        dt=self.param.Tf/self.param.nb_samples
        
        # Integrate position (q)
        q_next = cpin.integrate(self.cmodel, cx[:self.nq], cx[self.nq:] *  dt)

        # Update velocity (dq) using Euler forward
        dq_next = cx[self.nq:] + cu * dt#self.acc(cx, cu[:self.nv], cu[self.nv:])#cu * self.param.dt
        
        # Create CasADi function for next state concatenation
        self.dyn_fun = casadi.Function('dyn_fun', [cx, cu], [casadi.vertcat(q_next, dq_next)])
        
  
    
    def solve_doc_acados(self,param):
          
        T=int(param.nb_samples)
        quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), np.deg2rad(0), 0)).coeffs()# set the model up rigth
        q0=pin.neutral(self.model) 
        q0[3:7]=quat
       
        
        #q0=param.qdi#
        
        if param.free_flyer:
            # calculate static initial conditions for external forces and moments
            pin.forwardKinematics(self.model, self.data, q0)
            pin.updateFramePlacements(self.model, self.data)
            
            tau_ff0=pin.rnea(self.model,self.data,q0,np.zeros(self.nv),np.zeros(self.nv)) 
            tau_ff0 = np.asarray(tau_ff0 )#.reshape(6,)
            force_ff = pin.Force(tau_ff0[:3].reshape(3,), tau_ff0[3:6].reshape(3,))
            F_world = self.data.oMi[0].act(force_ff)
        
            F_lf = self.M_lf.inverse().actInv(F_world/2) # transforms F_world from world into the left foot frame.
            F_rf = self.M_rf.inverse().actInv(F_world/2) # transforms F_world from world into the right foot frame.
 
        
      
        # * Define the problem dimensions
        # define a state x = (q, v)^T to describe the robot dynamics
        nq=self.nq
        nv=self.nv
        nx =  nq +  nv     # state dimension
        nu = nv           # control
        nf = 12           # Got two 6D forces at the feet
        
       # -------------------------------------
        # ACADOS Optimal Control Problem (OCP)
        # -------------------------------------
        ocp = AcadosOcp()

        
        cx = casadi.SX.sym("cx", nx, 1) # state: the positions and velocities
        cu = casadi.SX.sym("cu", nu + nf, 1) # control: the torques and the two external wrenches
        
        
        ocp.model.disc_dyn_expr = self.dyn_fun(cx, cu[:nv])
        ocp.model.dyn_type = "discrete"
        
        # Solve
        ocp.solver_options.integrator_type = "DISCRETE"
        
        ocp.model.p = []

        ocp.model.name = "human_turbo_doc"
        ocp.model.x = cx
        ocp.model.u = cu


        # # Time horizon
        ocp.solver_options.N_horizon = self.param.nb_samples
        ocp.solver_options.tf = param.Tf


        ## path cost
        ocp.cost.cost_type = 'NONLINEAR_LS'
        
        cost_terms = []

        # --- base/general costs ---
        # cost_terms += [
        #     self.tau(cx, cu[:nv], cu[nv:]), # min joint torques
        #     self.com(cx, cu[:nv])[0],       # min COM X
        #     self.vcom(cx, cu[:nv]),         # min COM velocity
        #     cx[:nq],                        # joint positions regularization
        #     cx[nq+6:],                      # joint velocity (excluding free-flyer)
        #     cu[6:nv],                       # joint acc (excluding free-flyer)

        # ]
        
        idx_s = 0 # index of the slicer used to retrive cost index  used to build slices in the same order we will stack cost_y_expr
        cost_terms=[]
        W_blocks=[]
        if "min_joint_torque" in param.active_costs:
            cost_terms.append( self.tau(cx, cu[:nv], cu[nv:])  )# min joint torques 
            nb_elements=nv 
            if param.variables_w==True: 
                W_blocks.append( param.weights["min_joint_torque"][0]*np.eye(nb_elements) ) # will be overwrite later just to avoid error
            else:
                W_blocks.append( param.weights["min_joint_torque"]*np.eye(nb_elements) ) 
        
            sl_tau   = slice(idx_s, idx_s+nb_elements);           idx_s += nb_elements      # tau
          
        if "min_com_deviation" in param.active_costs:
            cost_terms.append( self.com(cx, cu[:nv])[0]  )# min com X
            nb_elements=1 
            if param.variables_w==True: 
                W_blocks.append( param.weights["min_com_deviation"][0]*np.eye(nb_elements) ) # will be overwrite later just to avoid error
            else:
                W_blocks.append( param.weights["min_com_deviation"]*np.eye(nb_elements) ) 
        
            sl_com   = slice(idx_s, idx_s+nb_elements);           idx_s += nb_elements      # com

        if "min_com_velocity" in param.active_costs:
            cost_terms.append( self.vcom(cx, cu[:nv])  )# min dcom  
            nb_elements=3 
            if param.variables_w==True: 
                W_blocks.append( param.weights["min_com_velocity"][0]*np.eye(nb_elements) ) # will be overwrite later just to avoid error
            else:
                W_blocks.append( param.weights["min_com_velocity"]*np.eye(nb_elements) ) 
        
            sl_dcom   = slice(idx_s, idx_s+nb_elements);           idx_s += nb_elements      # dcom

        
        if "min_joint_acc" in param.active_costs:
            cost_terms.append(  cu[6:nv]  )# min joint acc  no Free flyer
            nb_elements=nv-6
            if param.variables_w==True: 
                W_blocks.append( param.weights["min_joint_acc"][0]*np.eye(nb_elements) ) # will be overwrite later just to avoid error
            else:
                W_blocks.append( param.weights["min_joint_acc"]*np.eye(nb_elements) ) 
        
            sl_ddq  = slice(idx_s, idx_s+nb_elements);           idx_s += nb_elements      # ddq
            
            
        if "min_joint_vel" in param.active_costs:
            cost_terms.append(  cx[nq+6:]  )# min joint vel  no Free flyer
            nb_elements=nv-6
            if param.variables_w==True: 
                W_blocks.append( param.weights["min_joint_vel"][0]*np.eye(nb_elements) ) # will be overwrite later just to avoid error
            else:
                W_blocks.append( param.weights["min_joint_vel"]*np.eye(nb_elements) ) 
        
            sl_dq  = slice(idx_s, idx_s+nb_elements);           idx_s += nb_elements      # dq   
        
        
        
        W_q = 5e-3*np.diag(np.ones(nq)) # joint position regularisation
        cost_terms.append(cx[:nq])
        W_blocks += [ W_q ]            # (nq,nq)
        nb_elements=nq
        idx_s += nb_elements 

       
        if param.free_flyer:
            cost_terms += [
                            cu[nv:],                        # external forces and moments  regularization   
                            self.acc_cstr_lf(cx, cu[:nv]),  # no cartesian acc left foot
                            self.acc_cstr_rf(cx, cu[:nv]),   # no cartesian acc right foot
                            self.pos_error_lf(cx,cu[:self.nv]), # no displacement  left foot
                            self.pos_error_rf(cx,cu[:self.nv]) # no displacement acc right foot
            ]
            
           # regularisation terms (those should not be modified)
            W_ef= 1e-3*np.eye(12) 
            W_contact_pos=1e1*np.eye(6)
            W_contact_acc=1e3*np.eye(6)
   
     
            W_blocks += [
                            W_ef,           # (12,12)
                            W_contact_acc,  # (6,6)
                            W_contact_acc,  # (6,6)
                            W_contact_pos,  # (6,6)
                            W_contact_pos,  # (6,6)
                ]
            sl_free_flyer = slice(idx_s, idx_s+12+2*6+2*6);       idx_s += 12+2*6+2*6
            idx_start_target=idx_s
   
         
      
    
        # --- costs to target are always at the end as they are of variable size---
        
        
        nb_targets = len(param.FOI_to_set_Id) 
        
        # add all targets, regardless of how many there are
        for i in range(nb_targets):
            block = self.cost_to_target(cx, cu[:nv])[i*6:(i+1)*6]
            if param.FOI_axes[i] == "x":
                ax_target=[0]
            if param.FOI_axes[i] == "y":
                ax_target=[1]
            if param.FOI_axes[i] == "z":
                ax_target=[2]
            if param.FOI_axes[i] == "xy":
                ax_target=[0,1]
            if param.FOI_axes[i] == "xz":
                ax_target=[0,2]
            if param.FOI_axes[i] == "yz":
                ax_target=[1,2]    
            if param.FOI_axes[i] == "xyz":
                ax_target=[0,1,2]    
                    
            # only position for last two (first 3 components)
            cost_terms.append(block[ax_target]) #  
             
            
            W_target =param.weights["target"][i] * np.eye(len(ax_target))
            W_blocks.append(W_target)
            
            #cost_terms.append(block[:3]) #  
        # works whether W_target is a list of (3,3) arrays or an ndarray (n,3,3)    
        
        #W_target = np.array([param.weights["target"][s] * np.eye(3) for s in range(nb_targets)])
        #W_blocks += list(W_target[:nb_targets])        
                
                
        ocp.model.cost_y_expr = casadi.vertcat(*cost_terms) 
        ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.size()[0] ) # yref is filled with zeros
  
 
        # build W (acados expects a numpy array)
        dm_blocks = [casadi.DM(np.atleast_2d(b)) for b in W_blocks]  # force 2-D
        ocp.cost.W = casadi.diagcat(*dm_blocks).full()   
        
        
        
        
        
        
        

    # # # terminal cost (recopy of path cost but we keep it like that for clarity)
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        cost_terms_e = []
        W_blocks_e   = []
        for i in range(nb_targets):
            block_i = self.cost_to_target(cx, cu[:nv])[i*6:(i+1)*6]
            if param.FOI_axes[i] == "x":
                ax_target=[0]
            if param.FOI_axes[i] == "y":
                ax_target=[1]
            if param.FOI_axes[i] == "z":
                ax_target=[2]
            if param.FOI_axes[i] == "xy":
                ax_target=[0,1]
            if param.FOI_axes[i] == "xz":
                ax_target=[0,2]
            if param.FOI_axes[i] == "yz":
                ax_target=[1,2]    
            if param.FOI_axes[i] == "xyz":
                ax_target=[0,1,2]    
                
            cost_terms_e.append(block_i[ax_target])
            W_target =param.weights["target"][i] * np.eye(len(ax_target))
            W_blocks_e.append(W_target)
            
        if param.free_flyer:
            cost_terms_e.append(self.pos_error_lf(cx, np.zeros(nv)) )  # 6
            cost_terms_e.append(self.pos_error_rf(cx,np.zeros(nv))  )  # 6
            W_blocks_e += [W_contact_pos, W_contact_pos]    # 6x6 each
            
        ocp.model.cost_y_expr_e = casadi.vertcat(*cost_terms_e)    
     
        
        ocp.cost.yref_e =  np.zeros(int(ocp.model.cost_y_expr_e.shape[0]) ) 
        
        dm_blocks_e = [casadi.DM(np.atleast_2d(B)) for B in W_blocks_e]  # ensure 2-D CasADi DM
        ocp.cost.W_e = casadi.diagcat(*dm_blocks_e).full()
 


        

        # # Initial guess      
        ocp.constraints.x0 = np.concatenate( [q0,np.zeros(self.nv)] )
        
        ## State bounds
        q_min = np.array(self.model.lowerPositionLimit[7:])
        q_max = np.array(self.model.upperPositionLimit[7:])
        # # # joint limits
        ocp.constraints.idxbx = np.arange(7, nq)         # indices in x to constrain in the state-vector
        ocp.constraints.lbx = np.array(q_min)
        ocp.constraints.ubx = np.array(q_max)

        ## control bounds
        # indices of vertical forces in u
        # idx_fy_lf = nv + 1         # [Fx,Fy,Fz,Mx,My,Mz] -> Fy is +1 and vertical in foot frame
        # idx_fy_rf = nv + 6 + 1     # second foot block

        # ocp.constraints.idxbu = np.array([idx_fy_lf, idx_fy_rf], dtype=int)
        # ocp.constraints.lbu   = np.array([0.0, 0.0])      # Fy >= 0
        # ocp.constraints.ubu   = np.array([5e3, 5e3])      # no upper boun
        
        if param.free_flyer:
        # dynamical consistency
            dyn_cons = 1*(self.lf_force_at_ff(cx, cu[:nv], cu[nv:nv+6]) + self.rf_force_at_ff( cx, cu[:nv], cu[nv+6:] ) - self.tau_free(cx, cu[:nv])[:6]) # 
            com_cons_x=self.com(cx,cu[:nv])[0:1]  # add constraint on x-component of CoM
            com_cons_y=self.com(cx,cu[:nv])[1]  # add constraint on x-component of CoM

            
            ocp.model.con_h_expr = casadi.vertcat(dyn_cons,
                                                 com_cons_x,
                                                 com_cons_y,
                                                  self.pos_error_lf(cx,cu[:self.nv]),#self.cost_to_target(cx,cu[:self.nv])[2*6:(2+1)*6],
                                                  self.pos_error_rf(cx,cu[:self.nv]) )
            
            # bounds (min, max)
            ocp.constraints.lh = np.concatenate([
                                                np.zeros(6) - 1e-2,   # bounds for dynamical consistency
                                               np.array([self.BoS_min[0]- 0.05]), # heel bound
                                                np.array([self.BoS_min[1]- 0.05]), # lateral bound
                                                np.zeros(6) - 1e-2,
                                                np.zeros(6) - 1e-2,
                                                ])
    
            ocp.constraints.uh = np.concatenate([
                                                np.zeros(6) + 1e-2,   # bounds for dynamical consistency
                                               np.array([self.BoS_max[0]  + 0.22] ),#  # toe bound
                                               np.array([self.BoS_max[1] + 0.05] ), # lateral bound
                                                np.zeros(6) + 1e-2,
                                                np.zeros(6) + 1e-2,
                                                ])  
            
            # h_fun = casadi.Function(
            # "h_fun",
            # [cx, cu],
            # [ocp.model.con_h_expr]
            # )   
              
        # epsilon = 0.01  # 0.5cm tolerance
        # ocp.model.nh_e = 3
        # ocp.model.con_h_expr_e =self.cost_to_target(cx,cu[:self.nv])[0:3] #root_joint at desired depth
        # ocp.constraints.lh_e = -epsilon * np.ones(3)
        # ocp.constraints.uh_e = +epsilon * np.ones(3)

        epsilon = 0.01  # 0.5cm tolerance
        ocp.model.nh_e = nv
        ocp.model.con_h_expr_e =cx[nq:] #root_joint at desired depth
        ocp.constraints.lh_e = -epsilon * np.ones(nv)
        ocp.constraints.uh_e = +epsilon * np.ones(nv)
        
      
     # set options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        #ocp.solver_options.qp_solver_cond_N = 5  # horizon is long and DOF is modest, partial condensing makes the QP smaller

        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        #ocp.solver_options.integrator_type = 'IRK'
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        #ocp.solver_options.ext_cost_num_hess = 1  # 1 = exact Hessian from CasADi
        # Relax line search to allow more aggressive steps
        #ocp.solver_options.line_search_use_sufficient_descent = 1

        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        ocp.solver_options.regularize_method = 'MIRROR' # if SQP tehn regularize the hessian vaialble are NO_REGULARIZE, MIRROR, PROJECT, PROJECT_REDUC_HESS, CONVEXIFY, GERSHGORIN_LEVENBERG_MARQUARDT.
        ocp.solver_options.reg_epsilon = 1e-4#1e-6 more regularization can help larger steps and fewer backtracks:
        
        # QP options
        ocp.solver_options.qp_tol_stat  = 1e-4
        ocp.solver_options.qp_tol_eq    = 1e-4
        ocp.solver_options.qp_tol_ineq  = 1e-4
        ocp.solver_options.qp_tol_comp  = 1e-6


        # NLP options  

        ocp.solver_options.nlp_solver_tol_stat = 1e-1 # KKT residual value: most important stopping criteria !
        ocp.solver_options.nlp_solver_tol_eq   = 1e-3
        ocp.solver_options.nlp_solver_tol_ineq = 1e-3  
        ocp.solver_options.qp_tol_comp  = 5e-6
        
        ocp.solver_options.nlp_solver_max_iter = 100
        #ocp.solver_options.qp_solver_iter_max = 10 
        ocp.solver_options.qp_solver_warm_start = 1
         
        
        ocp.solver_options.print_level = 1

        # Create solver
        ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")#, build=False, generate=False)

        W_base = ocp.cost.W.copy()
        N = ocp.solver_options.N_horizon
        
        
        
        
        # --- windowed weights (exclude targets) ---
        # build slices for non-target blocks in the same order cost_y_expr was stacked

        nb_w = int(param.nb_w)
        for k in range(N):
            Wk = W_base.copy()

            # window index (equal partitions)
            wi = min(nb_w-1, (k * nb_w) // N)

            # helper: set scalar weight * I for a given slice
            # (kept inline, no function)
            # min_com_deviation -> COM x (1x1)
            if "min_com_deviation" in param.weights:
                val = param.weights["min_com_deviation"]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == nb_w:
                    val = val[wi]
                Wk[sl_com, sl_com] = float(val) * np.eye(1)

            # min_com_velocity -> vcom (3x3)
            if "min_com_velocity" in param.weights:
                val = param.weights["min_com_velocity"]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == nb_w:
                    val = val[wi]
                Wk[sl_dcom, sl_dcom] = float(val) * np.eye(3)

            # min_joint_vel -> dq (nv-6)
            if "min_joint_vel" in param.weights:
                val = param.weights["min_joint_vel"]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == nb_w:
                    val = val[wi]
                Wk[sl_dq, sl_dq] = float(val) * np.eye(sl_dq.stop - sl_dq.start)

            # min_joint_acc -> ddq (nv-6)
            if "min_joint_acc" in param.weights:
                val = param.weights["min_joint_acc"]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == nb_w:
                    val = val[wi]
                Wk[sl_ddq, sl_ddq] = float(val) * np.eye(sl_ddq.stop - sl_ddq.start)

            # min_joint_torque -> tau (nv)
            if "min_joint_torque" in param.weights:
                val = param.weights["min_joint_torque"]
                if isinstance(val, (list, tuple, np.ndarray)) and len(val) == nb_w:
                    val = val[wi]
                Wk[sl_tau, sl_tau] = float(val) * np.eye(nv)
        
         
            ocp_solver.cost_set(k, "W", Wk) # this is to avoid recompile
        
        
        
        
        
        
        
        # --- update target weights at runtime (no recompile) ---
 
        target_slices=[]
        start=idx_start_target
        for i in range(nb_targets):
            
            if param.FOI_axes[i] == "x":
                ax_target=[0]
            if param.FOI_axes[i] == "y":
                ax_target=[1]
            if param.FOI_axes[i] == "z":
                ax_target=[2]
            if param.FOI_axes[i] == "xy":
                ax_target=[0,1]
            if param.FOI_axes[i] == "xz":
                ax_target=[0,2]
            if param.FOI_axes[i] == "yz":
                ax_target=[1,2]    
            if param.FOI_axes[i] == "xyz":
                ax_target=[0,1,2]    
            
            s = len(ax_target)                  # 1, 2, or 3
            target_slices.append(slice(start, start + s))
            start += s   
             
 
        
        alpha = np.linspace(1e-3, 1.0, N)                 # linear ramp for weigths

        for k in range(N):
            Wk = W_base.copy()
            for sl in target_slices:
                base = W_base[sl, sl]            # len(ax)xlen(ax) per-target base weight
                Wk[sl, sl] = alpha[k] * base     # scale for this stage
            
                
            ocp_solver.cost_set(k, "W", Wk)
        
        
        # --- update end stage weights at runtime (no recompile) ---
        
        
        dm_blocks_e = [casadi.DM(np.atleast_2d(B)) for B in W_blocks_e]  # ensure 2-D CasADi DM
        W_e = casadi.diagcat(*dm_blocks_e).full()
        
        ocp_solver.cost_set(N, "W", W_e)
        
        
        # names in the exact stacking order used by cost_to_target
        names = [self.model.frames[i].name for i in param.FOI_to_set_Id]
        offset = nv + 1 + 3 + nq + (nv-6) + (nv-6) + ([12+6+6] if param.free_flyer else [0])[0]
        for i in range(nb_targets):
            sl = slice(offset + 3*i, offset + 3*(i+1))
            wdiag = np.diag(ocp.cost.W)[sl]
            print(f"target {i}: {names[i]}  diag(W)={wdiag}")
        
        
        t0 = time.time()
        for k in range(self.param.nb_samples-1):
             
            ocp_solver.set(k, "u", np.concatenate([np.zeros(self.nv), F_lf.linear,F_lf.angular,F_rf.linear,F_rf.angular]))
        
        
        for repeat_solve in range(1):
            print("iter:")
            print(repeat_solve)
            
            status = ocp_solver.solve()
        
        
            for k in range(self.param.nb_samples):
                ocp_solver.set(k, "x", ocp_solver.get(k+1, "x"))
                if k<self.param.nb_samples-1:
                    ocp_solver.set(k, "u", ocp_solver.get(k+1, "u"))
        
        
        elapsed = time.time() - t0
        print(f"Solve time: {elapsed*1e3:.3f} ms")

        
     
        
        if status != 0:
            print("❌ Solver failed with status:", status)
        else:
            print("✅ Solver succeeded")


            simX = np.zeros((self.param.nb_samples+1, nx))
            simU = np.zeros((self.param.nb_samples, nu+12))
            # get solution
            for i in range(self.param.nb_samples):
                simX[i,:] = ocp_solver.get(i, "x")
                simU[i,:] = ocp_solver.get(i, "u")
            simX[self.param.nb_samples,:] = ocp_solver.get(self.param.nb_samples, "x")
        
            
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, simX[i,:nq])  
            pin.updateFramePlacements(self.model, self.data)                   

            
            # h_vals = np.array([h_fun(simX[k], simU[k]).full().squeeze() 
            #        for k in range(len(simU))])

            # plt.plot(h_vals)
            # plt.title("h values over trajectory")
            # plt.show()
            
            
        ########## ADD This to a plot function
            
            # joint_indices = np.arange(7,  nq)  # e.g., joints excluding free-flyer
            # joint_names=[]
            # for idx in range(2,nq-5):
               
            #     joint_names.append(self.model.names[idx] )
            
            # #joint_names=joint_names[1:]
           
            # t = np.arange(simX.shape[0])  # discrete time steps
            
            
            # fig, axes = plt.subplots(int(len(joint_names)/2), 3, figsize=(6, 0.8*(len(joint_names))), sharex=True)
            # axes = axes.flatten()
            
            # for i, ax in enumerate(axes):  # start=1 to skip universe
            #     if i < nq-7:
            #         q_i = simX[:,i+7]

            #         ax.plot(q_i, 'r-', label="trajectory")
            #         ax.axhline(q_min[i], color="k", linestyle="--", label="q_min" if i == 1 else "")
            #         ax.axhline(q_max[i], color="k", linestyle="--", label="q_max" if i == 1 else "")
            #         ax.plot(t[0], q0[i+7], "kx", markersize=10, mew=2)
            #         ax.set_ylabel(joint_names[i])
            #         ax.grid(True)

            # axes[-1].set_xlabel("time step")
            # axes[0].legend()
            # plt.tight_layout()
            # plt.show()  
            
            
        ##########                     
                    
            actual_pose = self.data.oMf[self.model.getFrameId("root_joint")].copy()  # Desired EE pose

            print("position error")
            print(self.param.FOI_position[0]-actual_pose.translation)

        u_sol=simU[:,:nu]
        fs_sol=simU[:,nu:]
        return simX, u_sol, fs_sol   
        
        
    
        
        
        
            
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
    
   
 

    
    
    
    
    
    
    
    
    
    
    
    