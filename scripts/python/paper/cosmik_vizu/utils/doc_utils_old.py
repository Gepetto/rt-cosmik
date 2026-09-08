import casadi
from pinocchio import casadi as cpin
import numpy as np
from scipy import signal
import pandas as pd
import time

import matplotlib.pyplot as plt
from matplotlib import cm
 



def solve_inner_optimization(traj_id, w_norm, model, param):
    #Define a function that solves a simple optimization problem using CasADi


    runningModels = CostsModelDoublePendulum(model,param) 

    # # calculate DOC and cost function values for a given traj
    param["nb_samples"]=param["nb_samples_list"][traj_id]
    param["pxf"]=param["pxf_list"][traj_id]
    param["qdi"]=param["qdi_list"][traj_id]
    param["qdf"]=param["qdf_list"][traj_id]
    
        
    doc_problem = DocDoublePendulum(model,w_norm,runningModels, param) 
    q_opt_s, dq_opt_s, ddq_opt_s=doc_problem.solve_doc(model,[], param)
 
    return traj_id, q_opt_s, dq_opt_s, ddq_opt_s




class CostsBiomechanicalModel:
    
    def __init__(self,model,param):
         
        dt=param["dt"]
        self.dt =  dt
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        self.nq = cmodel.nq 
        self.nv = cmodel.nv 

        self.FOI_to_set=param["FOI_to_set"]
        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        # Casadi symbolics
        q = casadi.SX.sym("q",self.nq,1) # q
        dq = casadi.SX.sym("dq",self.nv,1) # dq
        ddq = casadi.SX.sym("ddq",self.nv,1) # ddq
        
        # Pinocchio all computations
        cpin.computeAllTerms(self.cmodel,self.cdata,q,dq)
        cpin.computeJointJacobians(self.cmodel,self.cdata,q)
        cpin.framesForwardKinematics(self.cmodel,self.cdata,q)
        
        


        
        
        
        
        
        dtau_dq, dtau_dv, dtau_da=cpin.computeRNEADerivatives(self.cmodel,self.cdata,q,dq,ddq)
        
        self.integrate = casadi.Function('integrate', [q, dq], [cpin.integrate(cmodel, q, dq*self.dt)])
        
        
        # Calculation of biomechanical cost
        # NOTE: All costs are normalized by user defined normalization (default is normalized by max feasible values from litterature)
        
        if param["free_flyer"]==True:
            self.tau=casadi.Function('tau',[q,dq,ddq],[cpin.rnea(cmodel,cdata,q,dq,ddq)[6:]/np.array(param["n_tau"]) ])

            self.dq_n=casadi.Function('dq_n',[q,dq,ddq],[dq[6:]/param["n_dq"] ])
            self.ddq_n=casadi.Function('ddq_n',[q,dq,ddq],[ ddq[6:]/param["n_ddq"] ])  
        else:
            self.tau=casadi.Function('tau',[q,dq,ddq],[cpin.rnea(cmodel,cdata,q,dq,ddq)/np.array(param["n_tau"]) ])
            self.dq_n=casadi.Function('dq_n',[q,dq,ddq],[dq/param["n_dq"] ])
            self.ddq_n=casadi.Function('ddq_n',[q,dq,ddq],[ ddq/param["n_ddq"] ])  
        
       # self.xdot = casadi.Function('xdot', [q,dq,ddq], [ casadi.vertcat(dq, ddq) ])  

        # CoP and CoM calculation
        # M_ankle = self.cdata.oMi[self.cmodel.getJointId('ankle_Z')]
        # ankle_wrench = M_ankle.act(self.cdata.f[self.cmodel.getJointId('ankle_Z')])
        # ankle_wrench_vector = ankle_wrench.vector
        # self.phi_ankle = casadi.Function('f', [q,dq,ddq], [ankle_wrench_vector])
        # self.cop = casadi.Function('cop', [q,dq,ddq], [casadi.vertcat(-ankle_wrench_vector[4]/ankle_wrench_vector[2],ankle_wrench_vector[3]/ankle_wrench_vector[2])]) # CoP
        self.com = casadi.Function('com', [q,dq,ddq], [self.cdata.com[0]]) # CoM
        self.vcom = casadi.Function('vcom', [q,dq,ddq], [self.cdata.vcom[0]]) # CoM velocity


        
      
        # Casadi Functions for cost function definition
        
       # self.energy=casadi.Function('energy',[q,dq,ddq],[ ( (dq[0]*self.tau(q,dq,ddq)[0])*(dq[0]*self.tau(q,dq,ddq)[0]) + (dq[1]*self.tau(q,dq,ddq)[1])*(dq[1]*self.tau(q,dq,ddq)[1]))  ])#[ (casadi.fabs (dq[0]*self.tau(q,dq,ddq)[0]) +casadi.fabs (dq[1]*self.tau(q,dq,ddq)[1]))  ])
        #self.geodesic=casadi.Function('geodesic',[q,dq,ddq],[ dq.T@cdata.M@dq   ])
       # self.tip = casadi.Function('tip', [q], [ self.cdata.oMf[-1].translation[[0,2]] ])
        #self.vtip =casadi.Function('vtip', [q,dq,ddq], [  cpin.getFrameVelocity(self.cmodel,self.cdata,cmodel.getFrameId('hand') ,cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[[0,2]] ] )
    
        
        frame_ids = []
        for frame_name in  self.FOI_to_set:
            if model.existFrame(frame_name):
                frame_id = model.getFrameId(frame_name)
                frame_ids.append(frame_id)
              
        self.position_frames=[]
        self.orientation_frames=[]
        for i in range(len (self.FOI_to_set)):
            self.position_frames.append(casadi.Function(f'tip_p_{frame_ids[i]}', [q], [ self.cdata.oMf[frame_ids[i]].translation ]))
            self.orientation_frames.append(casadi.Function(f'tip_r_{frame_ids[i]}', [q], [  casadi.SX(self.cdata.oMf[frame_ids[i]].rotation) ]))

 
        
        
    def calc(self,q,dq,ddq):
        # qnext=q+dq*self.dt
        # dqnext=dq+ddq*self.dt
        # Euler integration
      
        qnext=self.integrate(q,dq )
        
        dqnext=dq+ddq*self.dt
    
        self.J=[]
        
        self.J.append(self.tau(q,dq,ddq).T@self.tau(q,dq,ddq)) # min torque #C1
        self.J.append(self.dq_n(q,dq,ddq).T@self.dq_n(q,dq,ddq)) # min joint velocity #C2
        self.J.append(self.ddq_n(q,dq,ddq).T@self.ddq_n(q,dq,ddq)) # min joint acc #C3
        # self.J.append( (self.vtip(q,dq,ddq)[0].T@self.vtip(q,dq,ddq)[0]+self.vtip(q,dq,ddq)[1].T@self.vtip(q,dq,ddq)[1])/10 ) # min cartesian vel #C4
        # self.J.append( (self.dtau(q,dq,ddq).T@self.dtau(q,dq,ddq))/100 ) # min torque change #C5
        # self.J.append((self.energy(q,dq,ddq))/10) # min energy #C6
        # self.J.append((self.geodesic(q,dq,ddq))/4) # min geodesic #C7
        
        
        
        return qnext,dqnext, self.J 
    
    
class DocHumanMotionGeneration:
    
    def __init__(self,model, runningModels,param):
        self.weights = param["weights"] 
        self.param = param 
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        self.nq = cmodel.nq 
        self.runningModels=runningModels
     
 
        # retrieve parameters from param dictionnary
        self.qdi = param["qdi"]
        
        self.FOI_to_set=param["FOI_to_set"]
        self.FOI_position=param["FOI_position"]
        self.FOI_orientation=param["FOI_orientation"]
        self.FOI_sample=param["FOI_sample"]
        
        self.nb_samples = param["nb_samples"]
        self.q_min = param["q_min"]
        self.q_max = param["q_max"]
        self.dq_lim = param["dq_lim"]
        self.nb_w = param["nb_w"]
        self.free_flyer = param["free_flyer"]
        self.q0 = param["q0"]
        self.dq0 = param["dq0"]
        self.ddq0 = param["ddq0"]
        self.cop_lim = param["cop_lim"] 
       
        
    def solve_doc(self,model, fext, param):
        
        # for DOC
        opti_doc = casadi.Opti()

        cdt = casadi.SX.sym('cdt') # Time step
        
        # Decision variables
        qs = [ opti_doc.variable(model.nq) for i in range(self.nb_samples) ]     # state variable
        dqs = [ opti_doc.variable(model.nv) for i in range(self.nb_samples) ]     # state variable


        
        nf = 6                # Got a single 6D force
        cf = casadi.SX.sym("f", nf, 1)

        
        # * Build contact forces list
        idLeft = self.cmodel.getFrameId("left_foot")# left foot
        print(idLeft)
        forces = [ cpin.Force.Zero() for _ in self.cmodel.joints]
        cframe = self.cmodel.frames[idLeft] 
        cpin.Force(cf[:3], cf[3:])
        forces[cframe.parentJoint] = cframe.placement.act(cpin.Force(cf[:3], cf[3:]))

        cforces = cpin.StdVec_Force()
        for f in forces:
            cforces.append(f)
        
        
        
        
        
        if param["optimal_control"]==1: 
            tau = [ opti_doc.variable(model.nv) for i in range(self.nb_samples) ] # control variable
            ddqs=[]
        else:
            ddqs = [ opti_doc.variable(model.nv) for i in range(self.nb_samples) ]     # control variable
            
        
    
         

        
        add_free_flyer=0 
        if param["free_flyer"]==True:
                add_free_flyer=7
        
        R = casadi.SX.sym('R', 3, 3)
        R_ref = casadi.SX.sym('R_ref', 3, 3)
   
        log3 = casadi.Function('log', [R, R_ref], [cpin.log3(R.T @ R_ref)])
        
        
        # Roll out loop, summing the integral cost and defining the shooting constraints.
        # set 
        #   - initial joint configuration 
        #   - joint bounds 
        
        total_weigthed_cost = 0
        t_prev=-1
        j=0
        for t in range(self.nb_samples):
            
            if param["optimal_control"]==1:
                qnext, dqnext, J, = self.runningModels.calc_tau(qs[t],dqs[t], tau[t], fext[t]) # r for residue
                ddqs.append(self.runningModels.ddq(qs[t],dqs[t],tau[t],fext[t][0].vector,fext[t][1].vector,fext[t][2].vector))
            else:
                qnext, dqnext, J, = self.runningModels.calc(qs[t],dqs[t], ddqs[t]) # r for residue
                if param["find_feasible_sol"]==True:
                    J= casadi.sumsqr(qnext-qs[t] )+casadi.sumsqr(dqnext-dqs[t])
                 
   
            cost=0
        
          
            for i in range(param["nb_cost"]):
                if param["variables_w"]==1: 
                    if int(self.nb_samples/self.nb_w)==(t-t_prev):
                        
                        t_prev=t
                        if j<self.nb_w-1:
                            j=j+1  
                   
                    cost +=  self.weights[i,j]*J[i]

                else:
                    cost +=  self.weights[i]*J[i] ######### !!!!!!!!!!!!!!! to be uncommented 
                    
            # constraints over Euler integration
            if t <self.nb_samples-1:   
                opti_doc.subject_to(qs[t + 1] == qnext )
                opti_doc.subject_to(dqs[t + 1] == dqnext )
            
            if param["free_flyer"]==True:
                # COM/COP constraint
                opti_doc.subject_to(opti_doc.bounded(self.cop_lim[0],self.runningModels.com(qs[t],dqs[t],ddqs[t])[0][0],self.cop_lim[1]))
            #    opti_doc.subject_to(opti_doc.bounded(self.cop_lim[0],self.runningModels.cop(qs[t],dqs[t],ddqs[t])[0][0],self.cop_lim[1])) # COP x to be added later 

            # Joint limitation constraints
                for ii in range(add_free_flyer,len(self.q_min)):
                         
                    opti_doc.subject_to(opti_doc.bounded(self.q_min[ii], qs[t+1][ii], self.q_max[ii])) # joint limit 
                    opti_doc.subject_to(opti_doc.bounded(self.dq_lim[ii], dqs[t+1][ii], self.dq_lim[ii])) # joint vel limit q2
            

            total_weigthed_cost += cost
        
        # Additional initial and terminal constraint
        #print("qdi =",  self.qdi)
        opti_doc.subject_to(qs[0] == param["qdi"]) # initial joint configuration
       
        opti_doc.subject_to(dqs[0] == np.zeros(model.nv) ) # initial velocity ==0
        
        opti_doc.subject_to(qs[-1] == param["qdf"]) # final joint configuration

        
        # opti_doc.subject_to(ddqs[0] == 0)  # ddq==0

        #opti_doc.subject_to(dqs[-1] == self.dqf)  # terminal value velocity==0
        #opti_doc.subject_to(dqs[-1] == [0,0])  # terminal value velocity==0
        #opti_doc.subject_to(ddqs[-1] == (dqs[-1]-dqs[-2])/param["dt"])  # terminal value acc is consistent (this is required for gradient calculation)
        #opti_doc.subject_to(ddqs[-1] == ddqs[-2]) 
        
     
         
        for i in range(len(  self.FOI_position)):
            
            if param["FOI_sample"][i]=="all":
                for all in range(param["nb_samples"]):
                   
                    opti_doc.subject_to(self.runningModels.position_frames[i](qs[all])==self.FOI_position[i])  
                    
                    # Compute logarithmic map (minimal representation, 3D error)
                    opti_doc.subject_to(   log3( self.runningModels.orientation_frames[i](qs[all]), self.FOI_orientation[i]) == 0)

            else:
                opti_doc.subject_to(self.runningModels.position_frames[i](qs[param["FOI_sample"][i]])==self.FOI_position[i])  
        
        
        
        
        ### SOLVE
        opti_doc.minimize(total_weigthed_cost)
 
        
        # Solver options
        opts = {
            "ipopt.print_level": 5,  # Suppress solver output
            "ipopt.sb": "yes",  # Suppress banner
            "ipopt.max_iter": 1000,  # Maximum iterations
            "ipopt.linear_solver": "mumps",  # Linear solver
            "print_time": 2,  # Print timing information
            "expand": True,  # Expand expressions for better performance
             "ipopt.hessian_approximation": "limited-memory",  # Hessian approximation
            "ipopt.tol": 1e-4,  # Overall tolerance
            "ipopt.constr_viol_tol": 1e-3,  # Constraint violation tolerance
            "ipopt.compl_inf_tol": 1e-4,  # Complementarity tolerance
            "ipopt.dual_inf_tol": 1e-4,  # Dual infeasibility tolerance
            "ipopt.acceptable_tol": 1e-4,  # Acceptable tolerance
            "ipopt.acceptable_constr_viol_tol": 1e-3  # Acceptable constraint violation tolerance
        }
        # ,"jit": False, "jit_options": jit_options}#{'ipopt.print_level': 0}#, 'linear_solver':'mumps'}#"expand":True, 'ipopt.hessian_approximation':'limited-memory'}
        

        opti_doc.solver("ipopt", opts) # set numerical backend
        
         
         
        for i in range(self.nb_samples):
            opti_doc.set_initial(qs[i][:],  self.q0[i])#self.qdi)
            opti_doc.set_initial(dqs[i][:],  self.dq0[i])#self.qdi)
            opti_doc.set_initial(ddqs[i][:],  self.ddq0[i])#self.qdi)
        
        start_time = time.perf_counter()
        opti_doc.solve_limited()
        
        # save optimal solution
        qs_sol = np.array([ opti_doc.value(q) for q in qs ])
        dqs_sol = np.array([ opti_doc.value(dq) for dq in dqs ])
        ddqs_sol = np.array([ opti_doc.value(ddq) for ddq in ddqs ])

        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")  
        
        # nlp_grad_f = sol.get_function('nlp_grad_f')
        # grad = nlp_grad_f(sol.x) 
        # print(grad)
        return qs_sol,dqs_sol,ddqs_sol   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class CostsModelDoublePendulum:
    
    def __init__(self,model,param):
        dt=param["dt"]
        self.dt =  dt
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        self.nq = cmodel.nq 
        
        if param["optimal_control"]==1:
            # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
            q = casadi.SX.sym("q",self.nq,1) # q
            dq = casadi.SX.sym("dq",self.nq,1) # dq
            tau = casadi.SX.sym("tau",self.nq,1)

            aba_variables = [q,dq,tau]
            xdot_variables = [casadi.vertcat(q,dq),tau]

            cfext=[] # external forces
            for i in range(model.njoints):
                fname = "f"+str(i)
                cf = casadi.SX.sym(fname,6,1)
                aba_variables.append(cf)
                xdot_variables.append(cf)
                cfext.append(cpin.Force(cf))


            self.ddq=casadi.Function('ddq', aba_variables, [cpin.aba(cmodel,cdata,q,dq,tau,cfext)] )
            self.dq_n=casadi.Function('dq_n',[q,dq,tau],[dq/np.array([5*np.pi,5*np.pi]) ])
            self.ddq_n=casadi.Function('ddq_n',aba_variables,[cpin.aba(cmodel,cdata,q,dq,tau,cfext)/np.array([10*np.pi**2,10*np.pi**2]) ])
            
            self.xdot = casadi.Function('xdot', xdot_variables, [ casadi.vertcat(dq, self.ddq(*aba_variables)) ])
            
            cpin.computeJointJacobians(self.cmodel,self.cdata,q)
            cpin.framesForwardKinematics(self.cmodel,self.cdata,q)
            
            dtau_dq, dtau_dv, dtau_da=cpin.computeRNEADerivatives(self.cmodel,self.cdata,q,dq,self.ddq(*aba_variables))
            
            # Casadi Functions for cost function definition
            self.dtau=casadi.Function('dtau',aba_variables,[ dtau_dq@dq    ])
            self.energy=casadi.Function('energy',[q,dq,tau],[  (dq[0]*tau[0])*(dq[0]*tau[0]) + (dq[1]*tau[1])*(dq[1]*tau[1])  ])#[ (casadi.fabs (dq[0]*self.tau(q,dq,ddq)[0]) +casadi.fabs (dq[1]*self.tau(q,dq,ddq)[1]))  ])
            self.geodesic=casadi.Function('geodesic',[q,dq,tau],[ dq.T@cdata.M@dq   ])
            self.tip = casadi.Function('tip', [q], [ self.cdata.oMf[-1].translation[[0,2]] ])
            self.vtip =casadi.Function('vtip', [q,dq,tau], [  cpin.getFrameVelocity(self.cmodel,self.cdata,cmodel.getFrameId('hand') ,cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[[0,2]] ] )
        
        else:
            # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
            # Casadi symbolics
            q = casadi.SX.sym("q",self.nq,1) # q
            dq = casadi.SX.sym("dq",self.nq,1) # dq
            ddq = casadi.SX.sym("ddq",self.nq,1) # ddq
            
            # Casadi Function for refining problem variables
            self.tau=casadi.Function('tau',[q,dq,ddq],[cpin.rnea(cmodel,cdata,q,dq,ddq)/np.array([92,77]) ])# 92 and 77 are joint torque limit for shoulder and elbow 
            self.dq_n=casadi.Function('dq_n',[q,dq,ddq],[dq/np.array([5*np.pi,5*np.pi]) ])
            self.ddq_n=casadi.Function('ddq_n',[q,dq,ddq],[ddq/np.array([10*np.pi**2,10*np.pi**2]) ])  
            
            self.xdot = casadi.Function('xdot', [q,dq,ddq], [ casadi.vertcat(dq, ddq) ])  

            cpin.computeJointJacobians(self.cmodel,self.cdata,q)
            cpin.framesForwardKinematics(self.cmodel,self.cdata,q)
            
            dtau_dq, dtau_dv, dtau_da=cpin.computeRNEADerivatives(self.cmodel,self.cdata,q,dq,ddq)
            
            # Casadi Functions for cost function definition
            self.dtau=casadi.Function('dtau',[q,dq,ddq],[ dtau_dq@dq    ])
            self.energy=casadi.Function('energy',[q,dq,ddq],[ ( (dq[0]*self.tau(q,dq,ddq)[0])*(dq[0]*self.tau(q,dq,ddq)[0]) + (dq[1]*self.tau(q,dq,ddq)[1])*(dq[1]*self.tau(q,dq,ddq)[1]))  ])#[ (casadi.fabs (dq[0]*self.tau(q,dq,ddq)[0]) +casadi.fabs (dq[1]*self.tau(q,dq,ddq)[1]))  ])
            self.geodesic=casadi.Function('geodesic',[q,dq,ddq],[ dq.T@cdata.M@dq   ])
            self.tip = casadi.Function('tip', [q], [ self.cdata.oMf[-1].translation[[0,2]] ])
            self.vtip =casadi.Function('vtip', [q,dq,ddq], [  cpin.getFrameVelocity(self.cmodel,self.cdata,cmodel.getFrameId('hand') ,cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[[0,2]] ] )
        
        
    def calc(self,q,dq,ddq):
        qnext=q+dq*self.dt
        dqnext=dq+ddq*self.dt

        self.J=[]
        
        self.J.append(self.tau(q,dq,ddq).T@self.tau(q,dq,ddq)) # min torque #C1
        self.J.append(self.dq_n(q,dq,ddq).T@self.dq_n(q,dq,ddq)) # min joint velocity #C2
        self.J.append(self.ddq_n(q,dq,ddq).T@self.ddq_n(q,dq,ddq)) # min joint acc #C3
        self.J.append( (self.vtip(q,dq,ddq)[0].T@self.vtip(q,dq,ddq)[0]+self.vtip(q,dq,ddq)[1].T@self.vtip(q,dq,ddq)[1])/10 ) # min cartesian vel #C4
        self.J.append( (self.dtau(q,dq,ddq).T@self.dtau(q,dq,ddq))/100 ) # min torque change #C5
        self.J.append((self.energy(q,dq,ddq))/10) # min energy #C6
        self.J.append((self.geodesic(q,dq,ddq))/4) # min geodesic #C7
        
        return qnext,dqnext, self.J 
    
    def calc_tau(self,q,dq,tau,fext):
        F_variables = [casadi.vertcat(q,dq),tau]
        ddq_variables = [q,dq,tau]
        for ii in range(len(fext)):
            f_ii = fext[ii]
            F_variables.append(f_ii.vector)
            ddq_variables.append(f_ii.vector)


        # Runge-Kutta 4 integration
        F = self.xdot; dt = self.dt
        x=casadi.vertcat(q,dq)
        k1 = F(*F_variables)

        F_variables[0] = x + dt/2*k1
        k2 = F(*F_variables)

        F_variables[0] = x + dt/2*k2
        k3 = F(*F_variables)

        F_variables[0] = x + dt*k3
        k4 = F(*F_variables)
        xnext = x + dt/6*(k1+2*k2+2*k3+k4)

        
        #qnext=xnext[:self.nq] #q+dq*self.dt
        qnext=q+dq*self.dt
        dqnext=dq+self.ddq(*ddq_variables)*self.dt # xnext[self.nq:]#dq+self.ddq(q,dq,tau) *self.dt

        self.J=[]
        
        self.J.append(tau.T@tau)# min torque #C1
        self.J.append(self.dq_n(q,dq,tau).T@self.dq_n(q,dq,tau))# min joint velocity #C2
        self.J.append(self.ddq_n(*ddq_variables).T@self.ddq_n(*ddq_variables)) # min joint acc #C3
        self.J.append(((self.vtip(q,dq,tau))[0].T@self.vtip(q,dq,tau)[0]+self.vtip(q,dq,tau)[1].T@self.vtip(q,dq,tau)[1])/10)# min cartesian vel #C4
        self.J.append(((self.dtau(*ddq_variables)).T@self.dtau(*ddq_variables))/100 )# min torque change #C5
        self.J.append((self.energy(q,dq,tau) )/10) # min energy #C6
        self.J.append((self.geodesic(q,dq,tau) )/4) # min geodesic #C7
        
        return qnext,dqnext, self.J 
    
class DocDoublePendulum:
    
    def __init__(self,model,weights, runningModels,param):
        self.weights = weights.copy()
        self.param = param 
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        self.nq = cmodel.nq 
        self.runningModels=runningModels
        self.qdi = param["qdi"]
       # self.qdf = param["qdf"]
        self.pxf = param["pxf"]
        #self.pyf = param["pyf"]
        self.nb_samples = param["nb_samples"]
        self.q_min = param["q_min"]
        self.q_max = param["q_max"]
        self.dq_lim = param["dq_lim"]
        self.nb_w = param["nb_w"]

    def solve_doc(self,model, fext, param):
        
        # for DOC
        opti_doc = casadi.Opti()
  
        # Decision variables
        qs = [ opti_doc.variable(model.nq) for i in range(self.nb_samples) ]     # state variable
        dqs = [ opti_doc.variable(model.nq) for i in range(self.nb_samples) ]     # state variable
       
        
        if param["optimal_control"]==1: 
            tau = [ opti_doc.variable(model.nq) for i in range(self.nb_samples) ] # control variable
            ddqs=[]
        else:
            ddqs = [ opti_doc.variable(model.nq) for i in range(self.nb_samples) ]     # control variable
            
        # Roll out loop, summing the integral cost and defining the shooting constraints.
        
        opti_doc.subject_to(opti_doc.bounded(self.q_min[0], qs[0][0], self.q_max[0])) # joint limit q1
        opti_doc.subject_to(opti_doc.bounded(self.q_min[1], qs[0][1], self.q_max[1])) # joint limit q2
        opti_doc.subject_to(opti_doc.bounded(self.dq_lim[0], dqs[0][0], self.dq_lim[1])) # joint vel limit q2
        opti_doc.subject_to(opti_doc.bounded(self.dq_lim[0], dqs[0][1], self.dq_lim[1])) # joint vel limit q2

        total_weigthed_cost = 0
        t_prev=-1
        j=0
        for t in range(self.nb_samples):
            
            if param["optimal_control"]==1:
                qnext, dqnext, J, = self.runningModels.calc_tau(qs[t],dqs[t], tau[t], fext[t]) # r for residue
                ddqs.append(self.runningModels.ddq(qs[t],dqs[t],tau[t],fext[t][0].vector,fext[t][1].vector,fext[t][2].vector))
            else:
                qnext, dqnext, J, = self.runningModels.calc(qs[t],dqs[t], ddqs[t]) # r for residue
             
            cost=0
        
           # time.sleep(1)
            for i in range(param["nb_cost"]):
                if param["variables_w"]==1: 
                    if int(self.nb_samples/self.nb_w)==(t-t_prev):
                        
                        t_prev=t
                        if j<self.nb_w-1:
                            j=j+1  
                   
                    cost +=  self.weights[i,j]*J[i]

                else:
                    cost +=  self.weights[i]*J[i]
                    
                # euler integration
            
            if t <self.nb_samples-1:   
                opti_doc.subject_to(qs[t + 1] == qnext )
                opti_doc.subject_to(dqs[t + 1] == dqnext )
            
                # joint limits 
                opti_doc.subject_to(opti_doc.bounded(self.q_min[0], qs[t+1][0], self.q_max[0])) # joint limit q1
                opti_doc.subject_to(opti_doc.bounded(self.q_min[1], qs[t+1][1], self.q_max[1])) # joint limit q2
                opti_doc.subject_to(opti_doc.bounded(self.dq_lim[0], dqs[t+1][0], self.dq_lim[1])) # joint vel limit q2
                opti_doc.subject_to(opti_doc.bounded(self.dq_lim[0], dqs[t+1][1], self.dq_lim[1])) # joint vel limit q2

            total_weigthed_cost += cost
        
        # Additional initial and terminal constraint
        #print("qdi =",  self.qdi)
        opti_doc.subject_to(qs[0] == self.qdi) # initial joint position
        #opti_doc.subject_to(qs[-1] == self.qdf) # initial joint position
        opti_doc.subject_to(dqs[0] == [0,0]) # initial velocity ==0
        # opti_doc.subject_to(ddqs[0] == 0)  # ddq==0

        #opti_doc.subject_to(dqs[-1] == self.dqf)  # terminal value velocity==0
        #opti_doc.subject_to(dqs[-1] == [0,0])  # terminal value velocity==0
        #opti_doc.subject_to(ddqs[-1] == (dqs[-1]-dqs[-2])/param["dt"])  # terminal value acc is consistent (this is required for gradient calculation)
        #opti_doc.subject_to(ddqs[-1] == ddqs[-2]) 
        
        #opti_doc.subject_to(runningModels.tip(qs[-1],dqs[-1],ddqs[-1])==[param["pxf"],0]) # tip of pendulum at given position
        #print("pxf =",  self.pxf)
        # print("pyf =",  self.pyf)
        opti_doc.subject_to(self.runningModels.tip(qs[-1])[0]==self.pxf) # tip of pendulum on X axis at given position
       # opti_doc.subject_to(self.runningModels.tip(qs[-1])[1]==self.pyf) # tip of pendulum on Y axis at given position        
       

        ### SOLVE
        opti_doc.minimize(total_weigthed_cost)
 
 
        # ipopt options       
        jit_options = {"flags": ["-O3"], "verbose": False,"compiler": "ccache gcc","temp_suffix":False,"cleanup":False}
        #options = {"jit":True,"compiler":"shell"}
        #options["jit_options"] = {"compiler": "ccache gcc", "verbose":True} 
        
        
        p_opts = {'ipopt.print_level':0 , 'print_time': 0, "expand":False}# ,"jit": False, "jit_options": jit_options}#{'ipopt.print_level': 0}#, 'linear_solver':'mumps'}#"expand":True, 'ipopt.hessian_approximation':'limited-memory'}
        s_opts = {"max_iter": 500}

        opti_doc.solver("ipopt", p_opts) # set numerical backend
        for i in range(self.nb_samples):
            opti_doc.set_initial(qs[i][:],self.qdi)
        
        
        start_time = time.perf_counter()
        sol = opti_doc.solve_limited()
        qs_sol = np.array([ opti_doc.value(q) for q in qs ])
        dqs_sol = np.array([ opti_doc.value(dq) for dq in dqs ])
        ddqs_sol = np.array([ opti_doc.value(ddq) for ddq in ddqs ])

        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")  
        
        # nlp_grad_f = sol.get_function('nlp_grad_f')
        # grad = nlp_grad_f(sol.x) 
        # print(grad)
        return qs_sol,dqs_sol,ddqs_sol



class GradientsDocDoublePendulum:
     ##########################################
     ###
     ### This class calculates symbolic the Gradients of OCP of the double pendulum doc
     ### inputs are an  the runningmodel of the ocp and the parameters 
     ### ouputs are : 
     ###            -df the gradients of cost functions relatively to the state variables 
     ###            df is of size Nbcost*Nbweigths x 3*Nbsamples
     ###             -dh the gradients of the equality constraints relatively to the state variables 
     ###             dh is of size Nbeqconstraints x 3*Nbsamples
     ##########################################
     
    def __init__(self,model,weights, runningModels,param):
        self.weights = weights.copy()
        self.param = param 
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        self.nq = cmodel.nq 
        self.runningModels=runningModels
        
    def calculate_gradients_doc(self,model, param):
        
 
        # Decision variables
        qs=casadi.SX.sym('qs',(2,param["nb_samples"]))
        dqs=casadi.SX.sym('dqs',(2,param["nb_samples"]))
        ddqs=casadi.SX.sym('ddqs',(2,param["nb_samples"]))
         
        ######## Evaluate the cost and constraints at each time step
        if param["variables_w"]==1: 
            total_f = [[0 for _ in range(param["nb_w"] )] for _ in range(7)]
            cost=0
            t_prev=-1
            j=0
        else:
            total_f = [0 for _ in range(param["nb_cost"])]
        
        for t in range(param["nb_samples"]):
             
            qnext, dqnext,f = self.runningModels.calc(qs[:,t],dqs[:,t], ddqs[:,t]) # f for residue of each cost function
        
            ###### COST f
            for i in range(param["nb_cost"]):
                if param["variables_w"]==1: 
                    if int(param["nb_samples"]/param["nb_w"])==(t-t_prev):
                        
                        t_prev=t
                        if j<param["nb_w"]-1:
                            j=j+1  
                 
                    total_f[i][j] +=  f[i]#self.weights[i,j]*f[i]

                else:
                    total_f[i]+=f[i]  
            
            
            
            #for i in range(param["nb_cost"]):
                
                #total_f[i]+=f[i]  
            
            
            
            ###### EQUALITY CONSTRAINTS
                # euler integration constraints  
            
            if t<param["nb_samples"]-1:   
                h_q_i=casadi.Function("h_q_i"+str(t),[qs ,dqs ,ddqs ],[qs[:,t + 1] -qnext[:]]) 
                h_dq_i=casadi.Function("h_dq_i"+str(t),[qs ,dqs ,ddqs ],[dqs[:,t + 1] -dqnext[:]]) 
                
                if t==0:      
                
                    dh_q_all=calculate_gradient_states('h_q_i'+str(t), h_q_i, qs, dqs, ddqs )
                    dh_dq_all=calculate_gradient_states('h_dq_i'+str(t), h_dq_i, qs, dqs, ddqs )
                else:
                    dh_q_all=casadi.vertcat(dh_q_all,calculate_gradient_states('h_q_i'+str(t), h_q_i, qs, dqs, ddqs ))
                    dh_dq_all=casadi.vertcat(dh_dq_all,calculate_gradient_states('h_dq_i'+str(t), h_dq_i, qs, dqs, ddqs ))
            
              
                
        if param["variables_w"]==1: 
             
             for i in range(param["nb_cost"]):
                for j in range(param["nb_w"]):
                    total_f_func=casadi.Function("total_J",[qs ,dqs ,ddqs ],[ total_f[i][j] ])   
            
                    if i==0 and j==0:
                        print("zero")
                        df_all=calculate_gradient_states('total_J_func', total_f_func, qs, dqs, ddqs )
                    else:
                        df_all=casadi.vertcat(df_all,calculate_gradient_states('total_J_func', total_f_func, qs, dqs, ddqs ))       
             
             
             
        else:
                         
            for i in range(param["nb_cost"]):
                
                total_f_func=casadi.Function("total_J",[qs ,dqs ,ddqs ],[ total_f[i] ])   
            
                if i==0:
                    df_all=calculate_gradient_states('total_J_func', total_f_func, qs, dqs, ddqs )
                else:
                    df_all=casadi.vertcat(df_all,calculate_gradient_states('total_J_func', total_f_func, qs, dqs, ddqs ))          
         
          # initial and final condition equality constraints
              
        h_q0=casadi.Function("h_q0",[qs ,dqs ,ddqs ],[qs[:,0] -param["qdi"]]) 
        h_dq0=casadi.Function("h_dq0",[qs ,dqs ,ddqs ],[dqs[:,0] ]) 
        h_dqf=casadi.Function("h_dqf",[qs ,dqs ,ddqs ],[dqs[:,-1] ]) 
        h_Pxf=casadi.Function("h_Pxf",[qs ,dqs ,ddqs ],[self.runningModels.tip(qs[:,-1])[:] -param["pxf"]]) 
        
        dh_q0  = calculate_gradient_states('h_q0', h_q0, qs, dqs, ddqs )
        dh_dq0 = calculate_gradient_states('h_q0', h_dq0, qs, dqs, ddqs )
        dh_dqf = calculate_gradient_states('h_dqf', h_dqf, qs, dqs, ddqs )          
        dh_Pxf = calculate_gradient_states('h_Pxf', h_Pxf, qs, dqs, ddqs )   
        
        df=casadi.Function("df",[qs ,dqs ,ddqs ],[df_all])
 
        dh =  casadi.Function('dC_q',[qs,dqs,ddqs],[casadi.vertcat(dh_q_all,dh_dq_all,dh_q0,dh_dq0, dh_dqf, dh_Pxf)  ])
        
         
        
        return   df, dh


# 3 DOFS PLANAR SQUAT
class CostsModel3Dofs:
    def __init__(self, model, dt):
        self.dt =  dt
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        self.nq = cmodel.nq 
        
        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        # Casadi symbolics
        q = casadi.SX.sym("q",self.nq,1) # q
        dq = casadi.SX.sym("dq",self.nq,1) # dq
        ddq = casadi.SX.sym("ddq",self.nq,1) # ddq

        # Pinocchio computations
        cpin.computeAllTerms(self.cmodel,self.cdata,q,dq)
        cpin.computeJointJacobians(self.cmodel,self.cdata,q) # Not needed normally
        cpin.framesForwardKinematics(self.cmodel,self.cdata,q) # Not needed normally
        dtau_dq, dtau_dv, dtau_da=cpin.computeRNEADerivatives(self.cmodel,self.cdata,q,dq,ddq)
        
        # Casadi Function for refining problem variables
        self.tau=casadi.Function('tau',[q,dq,ddq],[cpin.rnea(cmodel,cdata,q,dq,ddq)])    # no normalisation for now /np.array([92,77]) ])# 92 and 77 are joint torque limit for shoulder and elbow 
        self.dq_n=casadi.Function('dq_n',[q,dq,ddq],[dq]) # no normalization for now /np.array([5*np.pi,5*np.pi]) ])
        self.ddq_n=casadi.Function('ddq_n',[q,dq,ddq],[ddq])# no normalisation for now  /np.array([10*np.pi**2,10*np.pi**2]) ])  
        
        self.xdot = casadi.Function('xdot', [q,dq,ddq], [ casadi.vertcat(dq, ddq) ])  
        
        # Casadi Functions for cost function definition
        self.dtau=casadi.Function('dtau',[q,dq,ddq],[ dtau_dq@dq    ])
        self.energy=casadi.Function('energy',[q,dq,ddq],[ ( (dq[0]*self.tau(q,dq,ddq)[0])*(dq[0]*self.tau(q,dq,ddq)[0]) + (dq[1]*self.tau(q,dq,ddq)[1])*(dq[1]*self.tau(q,dq,ddq)[1]))  ])#[ (casadi.fabs (dq[0]*self.tau(q,dq,ddq)[0]) +casadi.fabs (dq[1]*self.tau(q,dq,ddq)[1]))  ])
        self.geodesic=casadi.Function('geodesic',[q,dq,ddq],[ dq.T@cdata.M@dq   ])
        self.tip = casadi.Function('tip', [q], [ self.cdata.oMf[cmodel.getFrameId('trunk')].translation])
        self.vtip =casadi.Function('vtip', [q,dq,ddq], [  cpin.getFrameVelocity(self.cmodel,self.cdata,cmodel.getFrameId('trunk') ,cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[[0,2]] ] )
        
        # CoP and CoM calculation
        M_ankle = self.cdata.oMi[self.cmodel.getJointId('ankle_Z')]
        ankle_wrench = M_ankle.act(self.cdata.f[self.cmodel.getJointId('ankle_Z')])
        ankle_wrench_vector = ankle_wrench.vector
        self.phi_ankle = casadi.Function('f', [q,dq,ddq], [ankle_wrench_vector])
        self.cop = casadi.Function('cop', [q,dq,ddq], [casadi.vertcat(-ankle_wrench_vector[4]/ankle_wrench_vector[2],ankle_wrench_vector[3]/ankle_wrench_vector[2])]) # CoP
        self.com = casadi.Function('com', [q,dq,ddq], [self.cdata.com[0]]) # CoM
        self.vcom = casadi.Function('vcom', [q,dq,ddq], [self.cdata.vcom[0]]) # CoM velocity

    def calc(self,q,dq,ddq):
        qnext=q+dq*self.dt
        dqnext=dq+ddq*self.dt

        self.J=[]
        
        self.J.append((self.tau(q,dq,ddq).T@self.tau(q,dq,ddq))/200000) # min torque #C1
        self.J.append((self.dq_n(q,dq,ddq).T@self.dq_n(q,dq,ddq))/10) # min joint velocity #C2
        self.J.append((self.ddq_n(q,dq,ddq).T@self.ddq_n(q,dq,ddq))/100) # min joint acc #C3
        self.J.append(((self.vtip(q,dq,ddq)[0].T@self.vtip(q,dq,ddq)[0]+self.vtip(q,dq,ddq)[1].T@self.vtip(q,dq,ddq)[1]))*10) # remove normalisation for now /10 ) # min cartesian vel #C4
        self.J.append( (self.dtau(q,dq,ddq).T@self.dtau(q,dq,ddq))/10000) # no normalization for now /100 ) # min torque change #C5
        self.J.append((self.energy(q,dq,ddq))/10000) # no normalization for now /10) # min energy #C6
        self.J.append((self.geodesic(q,dq,ddq))/10) # no normalization for now /4) # min geodesic #C7
        # self.J.append((self.com(q,dq,ddq)[2].T@self.com(q,dq,ddq)[2])) # minimisation of com on the vertical axis 
        
        return qnext,dqnext, self.J 

class Doc3DofsSquat:
    def __init__(self,
                 model,
                 weights, 
                 runningModels,
                 param):
        self.model = model
        self.weights = weights.copy()
        self.param = param 
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cmodel.createData()
        self.nq = cmodel.nq 
        self.runningModels=runningModels
        self.cop_lim = param["cop_lim"] 
        self.pzf = param["pzf"]
        self.t_end_squat = param["t_end_squat"]
        self.nb_samples = param["nb_samples"]
        self.q_min = param["q_min"]
        self.q_max = param["q_max"]
        self.tau_lim = param["tau_lim"]
        self.dq_lim = param["dq_lim"]
        self.nb_w = param["nb_w"]
        self.nb_cost = param["nb_cost"]
        self.variable_w = param["variables_w"]

    def solve_doc(self, qdi, qdf):
        # for DOC
        opti_doc = casadi.Opti()
  
        # Decision variables
        qs = [ opti_doc.variable(self.model.nq) for _ in range(self.nb_samples) ]     # state variable
        dqs = [ opti_doc.variable(self.model.nq) for _ in range(self.nb_samples) ]     # state variable
        ddqs = [ opti_doc.variable(self.model.nq) for _ in range(self.nb_samples) ]     # control variable
            
        # Roll out loop, summing the integral cost and defining the shooting constraints.
        total_weigthed_cost = 0
        t_prev=-1
        j=0
        for t in range(self.nb_samples):
            qnext, dqnext, J, = self.runningModels.calc(qs[t],dqs[t], ddqs[t]) # r for residue
            cost=0
            for i in range(self.nb_cost):
                if self.variable_w ==1: 
                    if int(self.nb_samples/self.nb_w)==(t-t_prev):
                        t_prev=t
                        if j<self.nb_w-1:
                            j=j+1  
                    cost +=  self.weights[i,j]*J[i]
                else:
                    cost +=  self.weights[i]*J[i]
                    
            # euler integration
            if t <self.nb_samples-1:   
                opti_doc.subject_to(qs[t + 1] == qnext )
                opti_doc.subject_to(dqs[t + 1] == dqnext )

            # COP constraint
            # opti_doc.subject_to(opti_doc.bounded(self.cop_lim[0],self.runningModels.cop(qs[t],dqs[t],ddqs[t])[0][0],self.cop_lim[1])) # COP x
            
            #Bounds
            for j in range(self.model.nq):
                opti_doc.subject_to(opti_doc.bounded(self.q_min[j], qs[t][j], self.q_max[j])) # joint limit
            for j in range(self.model.nv):
                opti_doc.subject_to(opti_doc.bounded(self.dq_lim[0], dqs[t][j], self.dq_lim[1])) # joint vel limit 
            for j in range(len(self.tau_lim)):
                opti_doc.subject_to(opti_doc.bounded(-self.tau_lim[j], self.runningModels.tau(qs[t],dqs[t],ddqs[t])[j], self.tau_lim[j])) # tau limit

            total_weigthed_cost += cost
        
        # Additional initial and terminal constraint
        print("qdi =",  qdi)
        opti_doc.subject_to(qs[0] == qdi) # initial joint position
        opti_doc.subject_to(qs[-1] == qdf) # final joint position
        opti_doc.subject_to(dqs[0] == np.zeros(self.model.nq)) # initial velocity ==0
        opti_doc.subject_to(dqs[-1] == 0)  # terminal value velocity==0
        
        # print("pzf =",  self.pzf)
        # opti_doc.subject_to(self.runningModels.tip(qs[61])[2]==self.pzf) 
        opti_doc.subject_to(self.runningModels.com(qs[self.t_end_squat],dqs[self.t_end_squat], ddqs[self.t_end_squat])[2]==self.pzf) 

        ### SOLVE
        opti_doc.minimize(total_weigthed_cost)
        
        # Solver options
        opts = {
            "ipopt.print_level": 5,  # Suppress solver output
            "ipopt.sb": "yes",  # Suppress banner
            "ipopt.max_iter": 1000,  # Maximum iterations
            "ipopt.linear_solver": "mumps",  # Linear solver
            "print_time": 1,  # Print timing information
            "expand": True,  # Expand expressions for better performance
            # "ipopt.hessian_approximation": "limited-memory",  # Hessian approximation
            "ipopt.tol": 1e-3,  # Overall tolerance
            "ipopt.constr_viol_tol": 1e-6,  # Constraint violation tolerance
            "ipopt.compl_inf_tol": 1e-6,  # Complementarity tolerance
            "ipopt.dual_inf_tol": 1e-6,  # Dual infeasibility tolerance
            "ipopt.acceptable_tol": 1e-3,  # Acceptable tolerance
            "ipopt.acceptable_constr_viol_tol": 1e-5  # Acceptable constraint violation tolerance
        }

        opti_doc.solver("ipopt", opts) # set numerical backend
        
        # Warm start with initial guess
        for i in range(self.nb_samples):
            opti_doc.set_initial(qs[i][:], qdi)
        
        start_time = time.perf_counter()
        sol = opti_doc.solve_limited()
        qs_sol = np.array([ opti_doc.value(q) for q in qs ])
        dqs_sol = np.array([ opti_doc.value(dq) for dq in dqs ])
        ddqs_sol = np.array([ opti_doc.value(ddq) for ddq in ddqs ])

        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")  
        
        return qs_sol, dqs_sol, ddqs_sol

# 5 DOFS PLANAR BOX LIFTING
class CostsModel5Dofs:
    def __init__(self, model, dt):
        self.dt =  dt
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        self.nq = cmodel.nq 
        
        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        # Casadi symbolics
        q = casadi.SX.sym("q",self.nq,1) # q
        dq = casadi.SX.sym("dq",self.nq,1) # dq
        ddq = casadi.SX.sym("ddq",self.nq,1) # ddq

        # Pinocchio computations
        cpin.computeAllTerms(self.cmodel,self.cdata,q,dq)
        cpin.computeJointJacobians(self.cmodel,self.cdata,q) # Not needed normally
        cpin.framesForwardKinematics(self.cmodel,self.cdata,q) # Not needed normally
        dtau_dq, dtau_dv, dtau_da=cpin.computeRNEADerivatives(self.cmodel,self.cdata,q,dq,ddq)
        
        # Casadi Function for refining problem variables
        self.tau=casadi.Function('tau',[q,dq,ddq],[cpin.rnea(cmodel,cdata,q,dq,ddq)])    # no normalisation for now /np.array([92,77]) ])# 92 and 77 are joint torque limit for shoulder and elbow 
        self.dq_n=casadi.Function('dq_n',[q,dq,ddq],[dq]) # no normalization for now /np.array([5*np.pi,5*np.pi]) ])
        self.ddq_n=casadi.Function('ddq_n',[q,dq,ddq],[ddq])# no normalisation for now  /np.array([10*np.pi**2,10*np.pi**2]) ])  
        
        self.xdot = casadi.Function('xdot', [q,dq,ddq], [ casadi.vertcat(dq, ddq) ])  
        
        # Casadi Functions for cost function definition
        self.dtau=casadi.Function('dtau',[q,dq,ddq],[ dtau_dq@dq    ])
        self.energy=casadi.Function('energy',[q,dq,ddq],[ ( (dq[0]*self.tau(q,dq,ddq)[0])*(dq[0]*self.tau(q,dq,ddq)[0]) + (dq[1]*self.tau(q,dq,ddq)[1])*(dq[1]*self.tau(q,dq,ddq)[1]))  ])#[ (casadi.fabs (dq[0]*self.tau(q,dq,ddq)[0]) +casadi.fabs (dq[1]*self.tau(q,dq,ddq)[1]))  ])
        self.geodesic=casadi.Function('geodesic',[q,dq,ddq],[ dq.T@cdata.M@dq   ])
        self.tip = casadi.Function('tip', [q], [ self.cdata.oMf[cmodel.getFrameId('trunk')].translation])
        self.vtip =casadi.Function('vtip', [q,dq,ddq], [  cpin.getFrameVelocity(self.cmodel,self.cdata,cmodel.getFrameId('trunk') ,cpin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear[[0,2]] ] )
        
        # CoP and CoM calculation
        M_ankle = self.cdata.oMi[self.cmodel.getJointId('ankle_Z')]
        ankle_wrench = M_ankle.act(self.cdata.f[self.cmodel.getJointId('ankle_Z')])
        ankle_wrench_vector = ankle_wrench.vector
        self.phi_ankle = casadi.Function('f', [q,dq,ddq], [ankle_wrench_vector])
        self.cop = casadi.Function('cop', [q,dq,ddq], [casadi.vertcat(-ankle_wrench_vector[4]/ankle_wrench_vector[2],ankle_wrench_vector[3]/ankle_wrench_vector[2])]) # CoP
        self.com = casadi.Function('com', [q,dq,ddq], [self.cdata.com[0]]) # CoM
        self.vcom = casadi.Function('vcom', [q,dq,ddq], [self.cdata.vcom[0]]) # CoM velocity

    def calc(self,q,dq,ddq):
        qnext=q+dq*self.dt
        dqnext=dq+ddq*self.dt

        self.J=[]
        
        self.J.append((self.tau(q,dq,ddq).T@self.tau(q,dq,ddq))/1) # min torque #C1
        self.J.append((self.dq_n(q,dq,ddq).T@self.dq_n(q,dq,ddq))/1) # min joint velocity #C2
        self.J.append((self.ddq_n(q,dq,ddq).T@self.ddq_n(q,dq,ddq))/1) # min joint acc #C3
        self.J.append(((self.vtip(q,dq,ddq)[0].T@self.vtip(q,dq,ddq)[0]+self.vtip(q,dq,ddq)[1].T@self.vtip(q,dq,ddq)[1]))*1) # remove normalisation for now /10 ) # min cartesian vel #C4
        self.J.append( (self.dtau(q,dq,ddq).T@self.dtau(q,dq,ddq))/1) # no normalization for now /100 ) # min torque change #C5
        self.J.append((self.energy(q,dq,ddq))/1) # no normalization for now /10) # min energy #C6
        self.J.append((self.geodesic(q,dq,ddq))/1) # no normalization for now /4) # min geodesic #C7
         
        return qnext,dqnext, self.J 
    

