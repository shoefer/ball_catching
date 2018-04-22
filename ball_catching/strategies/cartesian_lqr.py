# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 21:39:42 2016

@author: Hoefer
"""

import os
import numpy as np
import matplotlib.pylab as plt
from numpy import isscalar

from filterpy.kalman import KalmanFilter#, ExtendedKalmanFilter
from filterpy.stats import plot_covariance_ellipse

from ball_catching.dynamics.world import Strategy, DynamicsModel
from ball_catching.utils.utils import make_hierarchical #, make_flat
from ball_catching.strategies.soc_solvers import LQR, iLQR, SOCBallCatching


def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))

# ----------------------------------------------------------------------------------------

class LQRStrategy(Strategy):
    # LQR solver - make it static so that we can 
    solver = None
  
    def __init__(self, dicts_, strategy_name="LQRStrategy"):
        super(LQRStrategy, self).__init__(dicts_)

        dicts = make_hierarchical(dicts_)
        
        # Costs
        self.terminal_distance = dicts[strategy_name]["terminal_distance"]
        self.terminal_velocity = dicts[strategy_name]["terminal_velocity"]
        self.control_effort = dicts[strategy_name]["control_effort"]
    
        self.use_kalman_filter = True
        try:
            self.use_kalman_filter = dicts[strategy_name]["use_kalman_filter"]
        except:
            pass

        self.use_lqr = True
        #self.use_lqr = False
    
        # KalmanFilter params
        self.kf_diag_min = 1e-15
        self.kf_P0_ball = dicts[strategy_name]["kf_P0_ball"]
        self.kf_P0_agent = dicts[strategy_name]["kf_P0_agent"]
        self.kf_P0_velocity_factor = dicts[strategy_name]["kf_P0_velocity_factor"]
        self.kf_Q_ball = dicts[strategy_name]["kf_Q_ball"]
        self.kf_Q_agent = dicts[strategy_name]["kf_Q_agent"]
        self.kf_Q_velocity_factor = dicts[strategy_name]["kf_Q_velocity_factor"]
        self.kf_R_ball = dicts[strategy_name]["kf_R_ball"]
        self.kf_R_agent = dicts[strategy_name]["kf_R_agent"]
        self.kf_R_velocity_factor = dicts[strategy_name]["kf_R_velocity_factor"]
        
    def initialize(self):
        self._init_dynamics()
        self._init_kalman_filter()
        self._init_lqr()

    def _init_dynamics(self):
        # Instantiate internal dynamics  
        dym_global = DynamicsModel()
        self.dynamics_local = DynamicsModel(dt=dym_global.DT, dim=dym_global.dim, 
                                      drag=False, copy=True)                  
        
    def _init_kalman_filter(self, filter_class=KalmanFilter):
        dynamics = self.dynamics_local
        self.kf = filter_class(dynamics.state_dim, dynamics.observation_dim, dynamics.action_dim)        
        
        # process model
        self.kf.F = dynamics.Adt
        self.kf.B = dynamics.Bdt
        # observation model
        self.kf.H = dynamics.Cdt
        
        # initial state covariance
        self.kf.P = np.identity(self.kf.P.shape[0]) * self.kf_diag_min
        for i in [0,3,6]:
          self.kf.P[i,i] = self.kf_P0_ball
        for i in [1,4,7]:
          self.kf.P[i,i] = self.kf_P0_ball*self.kf_P0_velocity_factor
          
        for i in [9,12]:
          self.kf.P[i,i] = self.kf_P0_agent
        for i in [10,13]:
          self.kf.P[i,i] = self.kf_P0_agent*self.kf_P0_velocity_factor
          
        self.P0 = np.zeros(self.kf.P.shape)
        self.P0[:] = self.kf.P
          
        # process noise covariance
        self.kf.Q = np.identity(self.kf.Q.shape[0]) * self.kf_diag_min
        for i in [0,3,6]:
          self.kf.Q[i,i] = self.kf_Q_ball
        for i in [1,4,7]:
          self.kf.Q[i,i] = self.kf_Q_ball*self.kf_Q_velocity_factor
          
        for i in [9,12]:
          self.kf.Q[i,i] = self.kf_Q_agent
        for i in [10,13]:
          self.kf.Q[i,i] = self.kf_Q_agent*self.kf_Q_velocity_factor
        
        # measurement noise covariance
        self.kf.R = np.identity(self.kf.R.shape[0]) * self.kf_diag_min
        for i in [0,1,2]:
          self.kf.R[i,i] = self.kf_R_ball
        for i in [3,5]:
          self.kf.R[i,i] = self.kf_R_agent
        for i in [4,6]:
          self.kf.R[i,i] = self.kf_R_agent*self.kf_R_velocity_factor
          
        self.kf.x[5] = -dynamics.GRAVITY
        #self.kf.test_matrix_dimensions()
          
    def _init_lqr(self, lqr_class=LQR):
        if not self.use_lqr:
            return
      
        lqr = lqr_class()      
        if self.__class__.solver is not None \
            and self.__class__.solver.solver.name == lqr.name \
            and self.__class__.solver.terminal_distance == self.terminal_distance \
            and self.__class__.solver.terminal_velocity  == self.terminal_velocity \
            and self.__class__.solver.control_effort == self.control_effort \
            and self.__class__.solver.solver.dynamics_local.is_equivalent(self.dynamics_local):
            self.solver = self.__class__.solver
            self.tti, self.F, self.f = self.solver.tti,\
              self.solver.F, self.solver.f
            return
        
        print ("[init_lqr] (Re-)Initializing control solver %s!" % lqr.name)        
        
        # Generating linear dynamics model without noise (copy)
        self.__class__.solver = SOCBallCatching(lqr,
                                      self.dynamics_local,
                                      self.terminal_distance,
                                      self.terminal_velocity,
                                      self.control_effort)
        
        result = self.__class__.solver.load()        

        self.solver = self.__class__.solver        
        self.tti, self.F, self.f = result

    def get_time_to_impact(self, x):
        return self.dynamics_local.get_time_to_impact(x)

    def find_current_gains(self, x):
        dynamics = self.dynamics_local
        fr = dynamics.FRAMERATE
                  
        t, N, x_n, z_n = self.get_time_to_impact(x)
        #t1,t2 = LQRStrategy.parabola_find_null_points(-g/2, x[4], x[3])
        #t = t1 if t1 > t2 else t2
        
        ret = None, None        
        
        if np.isnan(t):
            step = -1
        else:
            step = int(round(t*fr))-1
        
        if step < 0:
            #print t, N, x_n, z_n
            #raise Exception ("step is negative - %d" % step)
            #print ("WARN: step is negative - %d" % step)
            return ret
        
        try:
            F = self.F[step]
        except IndexError:
            #raise Exception ("%d is not a key in self.F" % step)
            #print("%d is not a key in self.F" % step)
            return ret
        try:
            f = self.f[step]
        except IndexError:
            raise Exception ("%d is not a key in self.f" % step)
            pass
        
        return F,f

    def start(self, **kwargs):
        if not self.use_kalman_filter:
            print (" WARN: kalman filter is off")
        if not self.use_lqr:
            print (" WARN: LQR is off")

        self._Z = []
        self._X = []
        self._X_obs = []
        self._U = []
        self._P = []
        
        self.initialize()

    def stop(self, **kwargs):
        if len (self._Z) > 0:
            self._X = np.array(self._X).reshape( (len(self._X),-1) )
            self._Z = np.array(self._Z).reshape( (len(self._Z),-1) )

        self._X_obs = np.array(self._X_obs).reshape( (len(self._X_obs),-1) )
        self._U = np.array(self._U).reshape( (len(self._U),-1) )

    def apply_filter(self, x):
        dynamics = self.dynamics_local

        # observe
        z = dynamics.Cdt.dot(x)
        
        if len(self._U) > 0:
          self.kf.predict(self._U[-1].reshape((-1,1)))

        # filter state
        self.kf.update(z.reshape((-1,1)))
        
        x_kf = self.kf.x
        
        self._Z.append(z)
        self._X.append(x_kf)
        self._P.append(self.kf.P)

        return x_kf
        
    def step(self, i, x, dicts):
        # -----------------------
        if self.use_kalman_filter:
            x_kf = self.apply_filter(x)
        else:  
            x_kf = x
        # -----------------------
            
        if not self.use_lqr:
            u = np.array([0,0])
            
        else:
            if self.use_kalman_filter and len(self._U) == 0:
              # in case we use KF, we wait one step till we 
              # get a somewhat useful state estimate
              u = np.array([0,0])
            else:
              F,f = self.find_current_gains(x_kf)
              
              if F is None:
                  u = np.array([0,0])
                  
              else:
                  assert (F is not None)
                  assert (f is not None)
                  
                  # old format: no accelerations in state, but bias term
                  if F.shape[1] == 11:
                      u = F.dot( np.concatenate( [x_kf.reshape(-1,)[[0,1, 3,4 ,6,7, 9,10, 12,13]] , [1]]))
                      u += f.reshape(u.shape)
                  # new format: accelerations in state
                  else:
                      assert (F.shape[1] == 15)
                      u = F.dot( x_kf )
                      u += f.reshape(u.shape)
                  u = u.reshape ((-1,))
                  #print ("u ", u)

        # logging
        self._U.append(u)
        self._X_obs.append(x)
        
        return u
  
    def plot_xy(self, fig=None):
        if fig is None:
            plt.figure(figsize=(10,10))
            
        for i in range(len(self._X)):
            _P = self._P[i]
            P = np.array([[_P[0,0], _P[0,3]],
                          [_P[3,0], _P[3,3]]])
            #print _P[:9,:9]
            plot_covariance_ellipse(self._X[i][[0,3]], cov=P)  
  
    def plot_dxy(self, fig=None):
        if fig is None:
            plt.figure(figsize=(10,10))
            
        for i in range(1,len(self._X)):
            _P = self._P[i]
            P = np.array([[_P[1,1], _P[1,4]],
                          [_P[4,1], _P[4,4]]])
            #print _P[:9,:9]
            plot_covariance_ellipse(self._X[i][[1,4]], cov=P)  
  

class ExtendProcessKalmanFilter(KalmanFilter):
    def predict(self, u, B=None, F=None, Q=None):
        assert(B is None)
        assert(F is None)
        
        dynamics = DynamicsModel()
        
        self.x = dynamics.step(self.x, u, noise=False)
        self.x = self.x.reshape( (-1,1) )
        
        # hmm, next x but previous u?
        F = dynamics.compute_J(self.x, u)

#        V = np.array(self.V_j.evalf(subs=self.subs)).astype(float)
#        # covariance of motion noise in control space
#        M = np.array([[self.std_vel*u[0]**2, 0], 
#                   [0, self.std_steer**2]])

        if Q is None:
            Q = self._Q
        elif isscalar(Q):
            Q = np.eye(self.dim_x) * Q
        
        self._P = self._alpha_sq * dot3(F, self._P, F.T) + Q

        #self.P = np.dot(F, self.P).np.dot(F.T) + np.dot(V, M).np.dot(V.T)


class iLQRStrategy(LQRStrategy):
    def __init__(self, dicts_):
        super(iLQRStrategy, self).__init__(dicts_, "LQRStrategy")
        
        #TODO expose drag settings
        self.drag_setting = dicts_["iLQRStrategy/drag_setting"]
        assert (self.drag_setting in ["baseball", "soccerball", ])
        
        self.use_kalman_filter = True
        try:
            self.use_kalman_filter = dicts_["iLQRStrategy/use_kalman_filter"]
        except:
            pass

        self.use_lqr = True

    def _init_dynamics(self):
        # Instantiate internal dynamics  
        dym_global = DynamicsModel()
        
        kwargs = {}
        if self.drag_setting == "baseball":
            # that's the default setting
            pass
        elif self.drag_setting == "soccerball":
            kwargs["mass"] = 0.4
            kwargs["c"] = 0.25
            kwargs["r"] = 0.11
          
        print kwargs
        self.dynamics_local = DynamicsModel(dt=dym_global.DT, dim=dym_global.dim, 
                                      drag=True, copy=True, **kwargs)
                                      
    def _init_kalman_filter(self, filter_class=ExtendProcessKalmanFilter):
        if filter_class==KalmanFilter:
            print ("[iLQRStrategy] WARNING using linear KalmanFilter")
        super(iLQRStrategy, self)._init_kalman_filter(filter_class)

    def _init_lqr(self, lqr_class=iLQR):
        super(iLQRStrategy, self)._init_lqr(lqr_class)

    def get_time_to_impact(self, x):
        # we ignore drag here because it gives us a massive speed-up in computation
        #return self.dynamics_local.get_time_to_impact(x, ignore_drag=True)
        return self.dynamics_local.get_time_to_impact(x, ignore_drag=False)

    def start(self, dicts=None):
        super(iLQRStrategy, self).start(dicts=dicts)
        
    def stop(self):
        super(iLQRStrategy, self).stop()

