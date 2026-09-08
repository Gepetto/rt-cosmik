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


def evaluate_costs(q,dq,ddq,runningModels, param):
    # this function evaluates all the cost functions in function of 
    # INPUTS: 
    #   - q, dq and ddq. These can be obtained from expert demonstrations or randomly generated.
    #   - runningModels is a class containing the calculation function for evaluating the cost individualy
    #   - param parameters dictionnary
    # OUTPUT: J_opt: a vector containing all the cost functions evaluated using the input trajectories
    
    # in case we have only one trajectory
    if param["nb_traj"]==1:
            param.update({
                 "nb_samples_list":[param["nb_samples"]]})
    
      
    if param["variables_w"]:
            J_opt=np.zeros(( param["nb_cost"], param["nb_w"], param["nb_traj"] ))  
            t_prev=0
            J_opt_temp=np.zeros(param["nb_cost"])
            k=0 
    else:
            J_opt=np.zeros((param["nb_cost"], param["nb_traj"]))

    for num in range(param["nb_traj"]):
            # Evaluate the optimal set of cost function
           

            k=0
            t_prev=0

            for t in range(param["nb_samples_list"][num]):
                if param["nb_traj"]==1:    
                    runningModels.calc(q[t,:],dq[t,:],ddq[t,:])
                else:
                    runningModels.calc(q[t,:,num],dq[t,:,num],ddq[t,:,num])
                    
                if param["variables_w"]:   
                    if int(param["nb_samples_list"][num]/param["nb_w"])==(t-t_prev):
                        t_prev=t
                        if k<param["nb_w"]-1:
                            k=k+1
                            J_opt_temp=np.zeros(param["nb_cost"])
                    
                    J_opt_temp+=np.array(runningModels.J ).reshape(param["nb_cost"]) 
                    
                    J_opt[:,k,num]=J_opt_temp 

                else:
                    J_opt[:,num]+=np.array(runningModels.J ).reshape(param["nb_cost"])


    if param["nb_traj"]==1:  
        J_opt=np.squeeze(J_opt)

    
    return J_opt

def IRL_step_calculation(w0, val_rmse_nopt, lambda_bretl, J_opt, J_nopt, param):
    p_opts = {'ipopt.print_level':5, 'print_time': 1, 'ipopt.sb': 'yes'}#{'ipopt.print_level': 0}#, 'linear_solver':'mumps'}#"expand":True}
    s_opts = {"max_iter": 10000}
    
    ## setup casadi optimisation problem to solve IRL 
    ## Similarly to Eq. from Kalakrishnan et al., Learning Objective Functions for Manipulation, ICRA 2013
    opti = casadi.Opti()
      
    if param["variables_w"]==1:
        w = casadi.SX.sym('w',(param["nb_cost"], param["nb_w"] ))
        x = opti.variable(param["nb_cost"], param["nb_w"] )
    else:
        w = casadi.SX.sym('w',param["nb_cost"])
        x = opti.variable(param["nb_cost"])  
    
    denominator_list = [] # denominator in the MaxENT function
    alpha=0
    
    if np.sum(param["best_weigths"])!=0:
        alpha=1
        
    if param['nb_traj']==1: # only one trajectory to fit
        for i in range(param["nb_nopt"]):
            Jopt_temp=0
            Jnopt_temp=0
            
            if param["variables_w"]:  
                for j in range(param["nb_cost"]):
                    for k in range( param["nb_w"] ):
                        Jopt_temp+=w[j,k]*J_opt[j,k]
                        Jnopt_temp+=w[j,k]*J_nopt[j,i,k]   
            else:
                for j in range(param["nb_cost"]):
                    Jopt_temp+=w[j]*J_opt[j]
                    Jnopt_temp+=w[j]*J_nopt[j,i]
            
            numerator=casadi.Function('numerator_i', [w], [casadi.exp(-1/lambda_bretl*Jopt_temp )])
            denominator_i=casadi.Function('denominator'+str(i), [w], [casadi.exp(-1/lambda_bretl*Jnopt_temp)])
            
            denominator_list.append(denominator_i)
        
        denominator_all=0    
        for i in range(param["nb_nopt"]):
            denominator_all+=denominator_list[i](x) #*((val_rmse_nopt[i])) 
            
        J_all= -casadi.log(numerator(x)/(denominator_all+1e-12)  )

    else: # more than one trajectory to fit
        denominator_i =[]
        denominator_list = [[None for _ in range(param["nb_traj"])] for _ in range(param["nb_nopt"])]
        numerator_list = []

        for num in range(param["nb_traj"]):
            for i in range(param["nb_nopt"]):   
                if param["variables_w"]:  
                    Jopt_temp=0
                    Jnopt_temp=0
                    for j in range(param["nb_cost"]):
                        for k in range( param["nb_w"] ):
                            if param["nb_traj"]==1:
                                Jopt_temp+=w[j,k]*J_opt[j,k]
                                Jnopt_temp+=w[j,k]*J_nopt[j,i,k]
                            else:
                                Jopt_temp+=w[j,k]*J_opt[j,k,num]
                                Jnopt_temp+=w[j,k]*J_nopt[j,i,k,num]
                else: # no variable weights 
                    Jopt_temp=0
                    Jnopt_temp=0
                    for j in range(param["nb_cost"]):                    
                        Jopt_temp+=w[j]*J_opt[j,num]
                        Jnopt_temp+=w[j]*J_nopt[j,i,num]

                numerator_num=casadi.Function('numerator_i', [w], [casadi.exp(-Jopt_temp )] )
                denominator_i=casadi.Function('denominator'+str(i), [w], [casadi.exp(  - Jnopt_temp )] )

                denominator_list[i][num]=denominator_i
            
            numerator_list.append(numerator_num)
            
    # setup IRL cost function (Kalakrishnan et al.,  2013)        
        for num in range(param["nb_traj"]):
            denominator_all=0    
            for i in range(param["nb_nopt"]):
                denominator_all+=denominator_list[i][num](x)#*((val_rmse_nopt[i,num])) 
            if num==0:
                J_all_temp= numerator_list[num](x)/denominator_all #/np.sum(val_rmse_nopt) 
            else:
                J_all_temp+=numerator_list[num](x)/denominator_all #/np.sum(val_rmse_nopt) 

        J_all= -casadi.log(J_all_temp )

    ## setup weight constraints and initial conditions
    if param["variables_w"]:  
        for i in range(param["nb_cost"]):
            for k in range( param["nb_w"] ):
                opti.subject_to(x[i,k]>0.00001)
                opti.subject_to(x[i,k]<10.99999)
           
                # if alpha==1:
                #     lower=param["best_weigths"][i,k]-0.1
                #     if lower<=0:
                #         lower=0.00001
                    
                #     upper=param["best_weigths"][i,k]+0.1

                #     #if upper<=0:
                #     #    upper=0.99999   
                        
                #     opti.subject_to(opti.bounded(lower, x[i,k], upper)) # joint limit q1
                # #alpha*casadi.sumsqr(param["best_weigths"]-x)
                
        cts=0       
        for k in range( param["nb_w"] ):
            # cts=0
            # for l in range(param["nb_cost"]):
            #     cts+=x[l,k]*x[l,k]
            # opti.subject_to(np.sqrt(cts)==1 )
            for l in range(param["nb_cost"]):
                cts+=x[l,k]
                
        # opti.subject_to(cts==1 )   
            
        for k in range( param["nb_w"] ):
            opti.set_initial(x[:,k], w0[:,k])
    
    else: # no variable weights 
        opti.subject_to(x>0.00001)
        opti.subject_to(x<0.99999)
        cts=0
        for i in range(param["nb_cost"]):
            cts+=x[i]*x[i]
        
        # opti.subject_to(np.sqrt(cts)==1 )    
        opti.set_initial(x,np.array(w0))
        
    ##Solve IRL using ipopt 
    opti.minimize( J_all )
    opti.solver("ipopt",p_opts, s_opts) 
    sol = opti.solve()
    x_opt=sol.value(x)        
    
    return x_opt

def rmse_function(a,b):
    n=len(a)
    y=np.sqrt( np.sum( (a-b)**2 )/n )
    
    return y


class CostsModelPolishing:
    
    def __init__(self,model,param):
        dt=param["dt"]
        self.dt =  dt
        self.cmodel = cmodel = cpin.Model(model)
        self.cdata = cdata = cmodel.createData()
        self.nq = cmodel.nq 
    
        # The self.xdot will be a casadi function mapping:  state,control -> [velocity,acceleration]
        # Casadi symbolics
        q = casadi.SX.sym("q",self.nq,1) # q
        dq = casadi.SX.sym("dq",self.nq,1) # dq
        ddq = casadi.SX.sym("ddq",self.nq,1) # ddq
        
        # Casadi Function for refining problem variables
        self.tau=casadi.Function('tau',[q,dq,ddq],[cpin.rnea(cmodel,cdata,q,dq,ddq) ])
      
    def calc(self,q,dq,ddq):
        qnext=q+dq*self.dt
        dqnext=dq+ddq*self.dt
    
        self.J=self.tau(q,dq,ddq).T@self.tau(q,dq,ddq) # min torque #C1
       
        return qnext,dqnext, self.J 
    

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
    


def cpin_ik_init(model,Pos_meas,q_init):
    #markers_names = ['RELB','RFIN']
  
    # Pos_meas = np.zeros(3*(len(markers_names)))

    # for ii in range(len(markers_names)):
    #     Pos_meas[ii]=DMarkers[markers_names[ii]][0]
    #     Pos_meas[len(markers_names)+ii]=DMarkers[markers_names[ii]][1]
    #     Pos_meas[2*len(markers_names)+ii]=DMarkers[markers_names[ii]][2]

    cq = casadi.SX.sym("cq",model.nq)

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    cpin.forwardKinematics(cmodel, cdata, cq)
    cpin.updateFramePlacements(cmodel,cdata)

    #Casadi Functions def 

    pos_RELB = casadi.Function('pos_RELB', [cq], [cdata.oMf[model.getFrameId("mk_elbow_est")].translation]) # RELB
    pos_RFIN = casadi.Function('pos_RFIN', [cq], [cdata.oMf[model.getFrameId("mk_hand_est")].translation]) # RFIN
    
    opti = casadi.Opti()
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes','ipopt.tol':1e-3}
    opti.solver('ipopt',opts)

    qs = opti.variable(model.nq)

    Posx_est=[]
    Posy_est=[]
    Posz_est=[]

    Posx_est = casadi.vertcat(Posx_est,pos_RELB(qs)[0])
    Posx_est = casadi.vertcat(Posx_est,pos_RFIN(qs)[0])

    Posy_est = casadi.vertcat(Posy_est,pos_RELB(qs)[1])
    Posy_est = casadi.vertcat(Posy_est,pos_RFIN(qs)[1])

    Posz_est = casadi.vertcat(Posz_est,pos_RELB(qs)[2])
    Posz_est = casadi.vertcat(Posz_est,pos_RFIN(qs)[2])

    Pos_est = casadi.vertcat(Posx_est,casadi.vertcat(Posy_est,Posz_est))

    cost = casadi.sumsqr(Pos_meas - Pos_est)
    opti.minimize(cost)
    opti.set_initial(qs,q_init)

   # opti.subject_to(casadi.sqrt(qs[3]*qs[3]+qs[4]*qs[4]+qs[5]*qs[5]+qs[6]*qs[6])==1) # Norm of quaternion must be unitary 
    opti.subject_to(opti.bounded(-np.pi,qs[0],np.pi))
    opti.subject_to(opti.bounded(-np.pi,qs[1],np.pi))
  



    sol=opti.solve()

    qs_sol = sol.value(qs)

    return qs_sol



# def load_csv(file_path):
#     # Skip the first 4 lines
#     # with open(file_path, 'r') as file:
#     #     for _ in range(4):
#     #         next(file)
#     # Load the rest of the file into a numpy array
#     print( tuple(range(18)))
#     data = np.genfromtxt(file_path, delimiter=',', skip_header=4,usecols=tuple(range(21)) )
#     return data


def load_csv(file_path):
    
    # Specify the number of rows to skip
    skiprows = 4
    # Load the CSV file into a DataFrame, skipping the specified number of rows
    df = pd.read_csv(file_path, skiprows=skiprows)
    # Now you can work with your DataFrame as needed
    # print(df.head())  # Display the first few rows of the DataFrame
    data = df.to_numpy()
    data = np.delete(data,0,axis = 1)
    
    return data
    



def low_pass_filter_data(data,dt,cut_off_frequency,nbutter):
    '''This function filters and elaborates data used in the identification process. 
    It is based on a return of experience  of Prof Maxime Gautier (LS2N, Nantes, France)'''
    
    b, a = signal.butter(nbutter, dt*cut_off_frequency/2, "low")
   
    data= signal.filtfilt(
            b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1) )
    
    
    # suppress end segments of samples due to the border effect
    nbord = 0#5 * nbutter
    data = np.delete(data, np.s_[0:nbord], axis=0)
    data = np.delete(data, np.s_[(data.shape[0] - nbord): data.shape[0]], axis=0)
     
    return data


def finite_difference_matrix(n, h):
    # Create an n x n matrix filled with zeros
    D = np.zeros((n, n))
    
    # Fill the matrix with the finite difference coefficients
    for i in range(n):
        D[i, i] = -2 / h**2  # Diagonal elements
        if i > 0:
            D[i, i-1] = 1 / h**2  # Below diagonal
        if i < n - 1:
            D[i, i+1] = 1 / h**2  # Above diagonal
    
    return D


def generate_non_optimal_traj(q_opt,dq_opt,ddq_opt, param):
    """_summary_
    Generate non optimal trajectories base on
        Kalakrishnan et al., base on STOMP: Stochastic Trajectory Optimization for Motion Planning, ICRA 2011.

    Args:
        q_opt (_type_): _description_
        dq_opt (_type_): _description_
        ddq_opt (_type_): _description_
        param (_type_): _description_

    Returns:
        a set of non optimal trajectories (position, velocity and acceleration)
    """
    
    q_nopt=np.empty( (param["nb_samples"],param["nb_joints"],param["nb_nopt"]) )
    dq_nopt=np.empty( (param["nb_samples"],param["nb_joints"],param["nb_nopt"]) )
    ddq_nopt=np.empty( (param["nb_samples"],param["nb_joints"],param["nb_nopt"]) )


    A=finite_difference_matrix(param["nb_samples"], 1)
    A0=np.zeros(param["nb_samples"])
    A0[0]=1
    A=np.vstack((A0,A))
    A0[0]=0
    A0[-1]=1

    A=np.vstack((A,A0))
    R=np.matmul(A.T,A)
    Rinv=np.linalg.inv(R)

    for i in range(param["nb_samples"]):
        Rinv[:,i]=Rinv[:,i]*1/(2*param["nb_samples"])
    
  

    covariance_matrix =  Rinv  
    
    # Perform Cholesky decomposition to correlate the samples
    L = np.linalg.cholesky(covariance_matrix.T)

    noise_std_small=param["noise_std"]/1

    for i in range(param["nb_nopt"]):
    
        # if i>param["nb_nopt"]/2:
        #      param["noise_std"]=noise_std_small
             
        for j in range(param["nb_joints"]):
           
            
            q_nopt[:,j,i]=q_opt[:,j]+ np.random.normal(0, 1*param["noise_std"], param["nb_samples"]) @ L.T # arbirtray choose parameters for noise 
           # q_nopt[:,j,i]=low_pass_filter_data(q_nopt[:,j,i],param["dt"],10,1)
           # dq_nopt[:,j,i]=np.hstack([np.diff(q_nopt[:,j,i])/param["dt"],dq_opt[-1,j]])
           # ddq_nopt[:,j,i]=np.hstack([np.diff(dq_nopt[:,j,i])/param["dt"],ddq_opt[-1,j]])
            dq_nopt[:,j,i]=dq_opt[:,j]+ np.random.normal(0, 1*(param["noise_std"]), param["nb_samples"]) @ L.T
            ddq_nopt[:,j,i]=ddq_opt[:,j]+ np.random.normal(0, 1*(param["noise_std"]), param["nb_samples"]) @ L.T

    return q_nopt, dq_nopt, ddq_nopt


def calculate_gradient_states(func_name,func, qs,dqs,ddqs):

 
    d_f_q = casadi.Function(func_name+'_q_i', [qs,dqs,ddqs],[ casadi.jacobian(func(qs,dqs,ddqs),qs)  ])
    d_f_dq = casadi.Function(func_name+'_dq_i', [qs,dqs,ddqs],[ casadi.jacobian(func(qs,dqs,ddqs),dqs)  ])
    d_f_ddq = casadi.Function(func_name+'_ddq_i', [qs,dqs,ddqs],[ casadi.jacobian(func(qs,dqs,ddqs),ddqs)  ])
    
    
    #for nb_eq in range(func.size_out(0)[0]):# get the number of equations to be derivated
    d_f=casadi.horzcat( casadi.horzcat( casadi.horzcat(d_f_q(qs,dqs,ddqs)[0,::2],d_f_q(qs,dqs,ddqs)[0,1::2]),  casadi.horzcat(d_f_dq(qs,dqs,ddqs)[0,::2],d_f_dq(qs,dqs,ddqs)[0,1::2])), casadi.horzcat(d_f_ddq(qs,dqs,ddqs)[0,::2],d_f_ddq(qs,dqs,ddqs)[0,1::2]) )
    
    #d_f=casadi.Function(func_name+'_q_testi'+str(t), [qs,dqs,ddqs],[ casadi.horzcat(d_f_q(qs,dqs,ddqs)[0,::2],d_f_q(qs,dqs,ddqs)[0,1::2])   ]) 
    
    for j in range(1,func.size_out(0)[0]):
         
        d_f= casadi.vertcat (d_f, casadi.horzcat( casadi.horzcat( casadi.horzcat(d_f_q(qs,dqs,ddqs)[j,::2],d_f_q(qs,dqs,ddqs)[j,1::2]),  casadi.horzcat(d_f_dq(qs,dqs,ddqs)[j,::2],d_f_dq(qs,dqs,ddqs)[j,1::2])), casadi.horzcat(d_f_ddq(qs,dqs,ddqs)[j,::2],d_f_ddq(qs,dqs,ddqs)[j,1::2]) )   )  
    
    return d_f


def plot_identified_weigths(w_norm,rmse, param):
    # Assuming w_norm is already defined and is of size 5x7
    # Example for illustration:
    # w_norm = np.random.rand(5, 7)  # Replace this with your actual w_norm data

    # Define the size of w_norm
    if param["variables_w"]==1:
        n_rows, n_cols = w_norm.shape
    else:
        n_rows=7
        n_cols = 1
        
    # Create the x and y coordinate arrays for the positions of the bars
    x = np.arange(n_rows)#.reshape(-1, 1)  # Row indices (reshaped to match the columns)
    y = np.arange(n_cols)  # Column indices

    # Create a meshgrid of x and y (positions of the bars)
    x, y = np.meshgrid(x, y, indexing="ij")

    # Flatten the arrays to plot them in 3D
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)  # Start at zero for the z-axis

    # The heights (dz) will be the values in w_norm, flattened
    dz = w_norm.flatten()

    # Set the width of the bars in x and y
    dx = dy = 0.5

    # Normalize the values in w_norm between 0 and 1 for color mapping
    norm = plt.Normalize(dz.min(), dz.max())
    colors = cm.RdYlGn(norm(dz))  # Red to Green colormap (reversed so 0 is green, 1 is red)

    # Create the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    
    # Plot the 3D bars with varying colors
    ax.bar3d(x, y, z, dx, dy, dz, color=colors, alpha=0.8)

    # Set custom x-axis labels
    x_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    ax.set_xticks(np.arange(n_rows))
    ax.set_xticklabels(x_labels)

    # Labels and title
    ax.set_xlabel('Cost labels')
    ax.set_ylabel('Windows index')
    ax.set_zlabel('Cost value')
    ax.set_title('Evolution of cost function values per windows index')
    text=str('RMSE:'+np.array2string(np.round(rmse, 4))+'deg' )
    ax.set_title(text)
    # Show the plot
    #plt.show()
