# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:09:12 2016

@author: Hoefer

0: t
1: x
2: y
3: z
4: vx
5: vy
6: vz
"""
from sys import path
#path.append(r"/home/belousov/local/casadi-py27-np1.9.1-v2.4.3")
path.append(r"/Users/Hoefer/Workspace/casadi-py27-np1.9.1-v2.4.3")
path.append(r"/home/shoefer/rrl_workspace/src/casadi-py27-np1.9.1-v2.4.3")

import casadi as ca

import pickle

from model import Model
from simulator import Simulator
from planner import Planner
import launcher

import numpy as np
import pandas as pd

import matplotlib.pylab as plt

import os

np.set_printoptions(suppress=True, precision=4)

#class Result:
#    def __init__(self)

class Embedded:
    
    def __init__(self, B, A = None, compute_plan=True):
        if B is None:
          self.use_B = False
          self.B = None
          self.dt = 0.1
        else:
          self.B = pd.DataFrame(data=B[:-1,:10], columns=[
            "t", "x", "z", "y", 
            "vx", "vz", "vy", # we switch z and y on purpose, z is up here
            "ax", "az", "ay",
            ])
          self.use_B = True
          self.dt = self.B.t.iloc[1]-self.B.t.iloc[0]
          print ("DT = %f" % self.dt)

        self.A = None
        if A is not None:
          self.A = pd.DataFrame(data=A[:-1,[0,1,3,4,6,7,9]], columns=[
            "t", "x", "y",
            "vx", "vy", # we switch z and y on purpose, z is up here
            "ax", "ay",
            ])
            
        self.compute_plan=compute_plan
        
        self.n_delay = 1
  
    def update_X(self, k, X):
        if not self.use_B:
          return X
        
        # x labels:
        #['x_b', 'y_b', 'z_b', 'vx_b', 'vy_b', 'vz_b', ]  (overwritten)
        # + ['x_c', 'y_c', 'vx_c', 'vy_c', 'phi', 'psi']
            
        X_ = X.cast()
        
        steps = X_.shape[1]
        for t in range(steps):
          #for i, col in enumerate(self.B.columns[1:7]):
          for i, col in enumerate(["x", "y","z","vx", "vy", "vz",]):
              X_[i,-t-1] = self.B[col].iloc[k-t]
          # FIXME always set agent velocity 0 !!!!!!!!!!! FIXME
          #X_[8,-t-1] = 0.
          #X_[9,-t-1] = 0.

        #return self.model.x.repeated(ca.horzcat(X_))
        return self.model.x.repeated(X_)
            
    def update_Z(self, k, Z):
        if not self.use_B:
          return Z

        # z labels:
        # ['x_b', 'y_b', 'z_b', ]  (overwritten)
        # + ['x_c', 'y_c', 'phi', 'psi']
        Z_ = Z.cast()
        
        steps = Z_.shape[1]
        for t in range(steps):
          for i, col in enumerate(["x","y","z"]):
              Z_[i,-t-1] = self.B[col].iloc[k-t]
        #return self.model.z.repeated(ca.horzcat(Z_))
        return self.model.z.repeated(Z_)
          
    def run(self):
        kwargs = {'dt': np.round(self.dt,5), 
                 'n_delay': self.n_delay,
#                   'F_c1': 7.5,
                 'F_c2': 7.5, # allow fast backwards running
                 'M_weight': 1e-15,
                 'N_min': 1e-15,
                 'N_max': 1e-15,
        }

        if self.use_B:
          kwargs.update({
                   'x_b0': self.B.iloc[0].x, 
                   'y_b0': self.B.iloc[0].y, 
                   'z_b0': self.B.iloc[0].z, 
                   'vx_b0': self.B.iloc[0].vx, 
                   'vy_b0': self.B.iloc[0].vy, 
                   'vz_b0': self.B.iloc[0].vz,          
          })

        print kwargs
      
        print ("Building models")
        self.model = launcher.new_model(**kwargs)
        self.model_p = launcher.new_model(**kwargs)
        
        print ("self.model.n_delay")
        print (self.model.n_delay)

        print ("Running MPC...")
        return self.mpc(self.model, self.model_p)
    
    def mpc(self, model, model_p): 
        
        # cls: simulate first n_delay time-steps with zero controls
        u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, model.n_delay))
        x_all = Simulator.simulate_trajectory(model, u_all)
        z_all = Simulator.simulate_observed_trajectory(model, x_all)
        b_all = Simulator.filter_observed_trajectory(model_p, z_all, u_all)
    
        #print x_all.cast()
        x_all = self.update_X(model.n_delay, x_all)
        #print x_all.cast()
        z_all = self.update_Z(model.n_delay, z_all)
    
        # Store simulation results
        X_all = x_all.cast()
        Z_all = z_all.cast()
        U_all = u_all.cast()
        B_all = b_all.cast()
    
        # Advance time
        model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])
    
        # Iterate until the ball hits the ground
        EB_all = []

        k = 0  # pointer to current catcher observation (= now - n_delay)

        while model.n != 0:
            rem = len(self.B)-k if self.use_B else model.n
            print ("k = %d (remaining: %d)" % (k, rem))
            
            if self.use_B and k+model.n_delay+1 >= len(self.B):
                print ("Recorded trajectory is over")
                break
            
            # Reaction delay compensation
            eb_all_head = Simulator.simulate_eb_trajectory(
                model_p,
                model_p.u.repeated(U_all[:, k:k+model_p.n_delay])
            )
            model_p.set_initial_state(
                eb_all_head[-1, 'm'],
                eb_all_head[-1, 'm'],
                eb_all_head[-1, 'L'] + eb_all_head[-1, 'S']
            )
            if model_p.n == 0:
                break
    
            # Planner: plan for model_p.n time steps
            if self.compute_plan:
              plan, lam_x, lam_g = Planner.create_plan(model_p)
              # plan, lam_x, lam_g = Planner.create_plan(
              #   model_p, warm_start=True,
              #   x0=plan, lam_x0=lam_x, lam_g0=lam_g
              # )
              belief_plan, _, _ = Planner.create_belief_plan(
                  model_p, warm_start=True,
                  x0=plan, lam_x0=lam_x, lam_g0=lam_g
              )
              u_all = model_p.u.repeated(ca.horzcat(belief_plan['U']))
            else:
              u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, model.n))
              #print model.n,model.n_delay,k
            
            # u_all = model_p.u.repeated(ca.horzcat(plan['U']))
    
            # cls: simulate ebelief trajectory for plotting
            #eb_all_tail = Simulator.simulate_eb_trajectory(model_p, u_all)
    
            # cls: execute the first action
            x_old = x_all
            x_all = Simulator.simulate_trajectory(model, [u_all[0]])

            self.apply_bc_dynamics(x_old[-1], u_all[0], x_all[-1])

            #print "----"
            #print x_all.cast()
            #try:
            x_all = self.update_X(k+model.n_delay+1, x_all)
            #x_all = self.update_X(k+x_all.shape[1], x_all)
#            except IndexError as e:
#              print ("ERROR")
#              print (e)
#              break
            #print x_all.cast()
            
            z_all = Simulator.simulate_observed_trajectory(model, x_all)
#            try:
            z_all = self.update_Z(k+model.n_delay+1, z_all)
            #z_all = self.update_Z(k+z_all.shape[2], z_all)
#            except IndexError as e:
#              print ("ERROR")
#              print (e)
#              break
            
            b_all = Simulator.filter_observed_trajectory(
                model_p, z_all, [u_all[0]]
            )
    
            # Save simulation results
            X_all.appendColumns(x_all.cast()[:, 1:])
            Z_all.appendColumns(z_all.cast()[:, 1:])
            U_all.appendColumns(u_all.cast()[:, 0])
            B_all.appendColumns(b_all.cast()[:, 1:])
            #EB_all.append([eb_all_head, eb_all_tail])
    
            # Advance time
            model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])
            model_p.set_initial_state(
                model_p.b(B_all[:, k+1])['m'],
                model_p.b(B_all[:, k+1])['m'],
                model_p.b(B_all[:, k+1])['S']
            )
            k += 1

            
            
        return X_all, Z_all, U_all, B_all, EB_all


    def apply_bc_dynamics(self, x, u, xn):
        pass
#        path.append(r"/home/shoefer/rrl_workspace/src/ball_catching/src")
#        from dynamics_simple import DynamicsModel
#        from dynamics_simple_mpc import MPCStrategy
#        DynamicsModel(dt=0.1)      
#        # testing
#        def pretty_xbc(xn_bc):
#          print "ball   x, y, z:    " + str(xn_bc[[0,3,6]])
#          print "ball  vx,vy,vz: " + str(xn_bc[[1,4,7]])
#          print "agent     x, z: " + str(xn_bc[[9,12]])
#          print "agent    vx,vz: " + str(xn_bc[[10,13]])
#          
#      
#        # COMPARE to our dynamics model
#        #u_bc = MPCStrategy._transform_U(x, u)
#        u_bc = map (lambda _u: _u/self.dt, [xn[8]-x[8], xn[9]-x[9]])
#        x_bc = MPCStrategy._x_to_xbc(x)
#        
#        xn_bc = DynamicsModel().step(x_bc, u_bc)
#        print "[Previous]"
#        pretty_xbc(x_bc)
#        print " u     -> " + str(u_bc)
#        print " u_raw -> " + str(map(float,u))
#        print "[BC   Model]"
#        pretty_xbc(xn_bc)
#        print "[Easy Model]"
#        pretty_xbc(self._x_to_xbc(xn))
#        print "(States raw)"
#        print map(float,x)
#        print map(float,xn)
#        #raw_input()
#        print "----------"      

class MyPlotter():
    @classmethod
    def plot_file(cls, fn):
      with open(fn, "r") as f:
        (X_all, Z_all, U_all, B_all, EB_all), (dfB, dfA) = pickle.load(f)
      cls.plot(os.path.basename(fn), X_all, dfB, dfA, os.path.basename(fn))
    
    
    @classmethod
    def plot(cls, title, X_all, dfB, dfA, fn_prefix=None):
      title = title.replace("_", " ")
      suffix = ".pdf"
      
      if False:
        # debugging plots
        plt.figure()
        plt.title(title)
        if dfB is not None:
          plt.plot(dfB.x, label="bc x", c="k")
        plt.plot(np.array(X_all[0,:]).T, label="easy x", c="k", ls="--")
        plt.legend(loc="lower right")
        
        plt.figure()
        plt.title(title)
        if dfB is not None:
          plt.plot(dfB.y, label="bc y", c="k")
        plt.plot(np.array(X_all[1,:]).T, label="easy y", c="k", ls="--")
        plt.legend(loc="lower right")
    
        plt.figure()
        plt.title(title)
        if dfB is not None:
          plt.plot(dfB.z, label="bc z", c="k")
        plt.plot(np.array(X_all[2,:]).T, label="easy z", c="k", ls="--")
        plt.legend()
  
      plt.figure()
      plt.title(title)
      if dfB is not None:
        plt.plot(dfB.x, dfB.z, label="bc x-z", c="k")
        plt.plot(dfB.y, dfB.z, label="bc y-z", c="b")
      plt.plot(np.array(X_all[0,:]).T, np.array(X_all[2,:]).T, label="MPC x-z", c="k", ls="--")
      plt.plot(np.array(X_all[1,:]).T, np.array(X_all[2,:]).T, label="MPC y-z", c="b", ls="--")
      plt.legend()
  
      # Distance
      mpc_xt = np.asarray(X_all.T)[-1]
      mpc_dist = np.linalg.norm( mpc_xt[[0,1,]] - mpc_xt[[6,7,]] )
      dist_str = "MPC: %.5f" % mpc_dist
      print (dist_str)

      fig = plt.figure(figsize=(10,4))
      if dfB is not None:
        plt.subplot(1,2,1)
      plt.title(title + " " + dist_str)
      plt.plot(np.array(X_all[0,:]).T, np.array(X_all[1,:]).T, label="MPC ball x-y", c="k", ls="--")
      plt.plot(np.array(X_all[6,:]).T, np.array(X_all[7,:]).T, label="MPC agent x-zy", c="b", )
      plt.legend(loc="lower right")

      if dfB is not None:
        oac_dist = np.linalg.norm(dfB.iloc[-2][["x", "y"]] - dfA.iloc[-2][["x", "y"]])
        dist_str = "OAC-CBA: %.5f" % oac_dist
        print (dist_str)
        
        #plt.figure()
        plt.subplot(1,2,2)
        plt.title(title + " " + dist_str)
        plt.plot(dfB.x, dfB.y, label="ball x-y", c="k", ls="--")
        plt.plot(dfA.x, dfA.y, label="agent x-y", c="b", )
        plt.legend(loc="lower right")
        #if fn_prefix is not None:
        #  plt.savefig(fn_prefix+" OAC-CBA"+suffix)
  
      if fn_prefix is not None:
        plt.savefig(fn_prefix+suffix)
      

def export_easy_catch_trajectory(fn, X_all, Z_all, dt):
    required_cols = ["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", ]
    df = pd.DataFrame(index = range(Z_all.shape[1]), columns = required_cols)
    for i in range(Z_all.shape[1]):
        df.iloc[i].t = float(i*dt)
        df.iloc[i].x = float(Z_all[0,i])
        df.iloc[i].z = float(Z_all[1,i]) # switch z and y (y is up in our simulator)
        df.iloc[i].y = float(Z_all[2,i])
        # compute velocity and acceleration
        if i > 0:
          df.iloc[i].vx = (df.iloc[i].x - df.iloc[i-1].x)/dt
          df.iloc[i].vy = (df.iloc[i].y - df.iloc[i-1].y)/dt
          df.iloc[i].vz = (df.iloc[i].z - df.iloc[i-1].z)/dt
          
        if i > 1:
          df.iloc[i].ax = (df.iloc[i].vx - df.iloc[i-1].vx)/dt
          df.iloc[i].ay = (df.iloc[i].vy - df.iloc[i-1].vy)/dt
          df.iloc[i].az = (df.iloc[i].vz - df.iloc[i-1].vz)/dt

    print ("Storing trajectory %s" % fn)
    df.to_pickle(fn)


if __name__ == "__main__":
  
    compute_plan = True
    #compute_plan = False  # testing

    files = {
      #'gaussian': None,
      #'drag3d': "/Users/Hoefer/Documents/ros/OAC3DStrategy_3d_drag_2016-08-15_22-32-45-558010/BC_OAC3DStrategy_2016-08-15_22-32-45-559292",
      'ideal3d': "/home/shoefer/.ros/data/MPCStrategy_2d_ideal_DTinv-10_2016-08-31_14-29-18-942881/BC_MPCStrategy_2016-08-31_14-29-18-943147/",
      ##'ideal2d': "/Users/Hoefer/Documents/ros/COVOAC2RawStrategyRIDGE_TAN_DOT_LIN_U_2d_ideal_2016-08-15_18-20-30-682297/BC_COVOAC2RawStrategy_2016-08-15_18-20-30-683985/ball_1.log"
      #'nathan': "/Users/Hoefer/Documents/ros/nathan01_COVOAC3DStrategy_3d_ideal_2016-08-17_11-26-19-425667/BC_COVOAC3DStrategy_2016-08-17_11-26-19-426804"

    }
    
    #for n_delay in [1,2,5]:
    #for n_delay in [5]:
    for n_delay in [1]:
  
      for nm, fn in files.items():
        
        A = None
        B = None
        if fn is not None:
          fn_ball = os.path.join(fn, "ball_1.log")
          fn_agent = os.path.join(fn, "agent_1.log")
          A = np.loadtxt(fn_agent)
          B = np.loadtxt(fn_ball)
        
        emb = Embedded(B, A, compute_plan=compute_plan)
        emb.n_delay = n_delay
        
        dfB = emb.B
        dfA = emb.A
    
        res = emb.run()
        X_all, Z_all, U_all, B_all, EB_all = res
        
        MyPlotter.plot(nm, X_all, dfB, dfA)
        
        if compute_plan:
          # store    
          with open("result_%s_nd%d.pkl" % (nm, n_delay), "w") as f:
            pkl = (X_all, Z_all, U_all, B_all, EB_all), (dfB, dfA)
            pickle.dump(pkl, f)
          
        export_easy_catch_trajectory("trajectory_%s_nd%d.pkl" % (nm, n_delay), X_all, Z_all, emb.model.dt)
        
    plt.show()