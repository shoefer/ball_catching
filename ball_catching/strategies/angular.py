# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:21:38 2016

@author: shoefer
"""

import numpy as np

from ball_catching.dynamics.world import observe_angle_and_distance, \
  compute_bearing_angle, Strategy, DynamicsModel
from ball_catching.strategies.utils import queue_push, compute_rv_index, windowed_averaged_dot

from ball_catching.utils.utils import make_hierarchical, make_flat

# ----------------------------------------------------------------------------------------


def compute_consistent_vref(nu, phi, D):
    g = 9.81
    R = nu**2 / g * np.sin(2*phi)
    return np.sin(phi)*nu/(R+D)

def compute_consistent_vref_from_dicts(dicts):
    b = dicts['Ball']
    a = dicts['Agent']

    assert (b['v_init_type'] == 'spherical')
    phi = b['v_init'][0]
    nu = b['v_init'][2]

    assert (a['x_init_type'] == 'cartesian')
    assert (a['x_relative'])
    D = a['x_init'][0]

    return compute_consistent_vref(nu, phi, D)

# ----------------------------------------------------------------------------------------

class COVOAC2DStrategy(Strategy):
  def __init__(self, dicts_, strategy_name = 'COVOAC2DStrategy'):
    
    if strategy_name == 'COVOAC2DStrategy':
      assert (DynamicsModel().dim == 2)
    
    self.ddtheta_min = 0.003
    #self.ddtheta_min = 0.0

    dicts = make_hierarchical(dicts_)

    # sanity check: is framerate as expected?
    self.framerate_rnm = 1.
    framerate = dicts['BallCatching']['framerate']
    framerate_expected = dicts[strategy_name]['framerate']
    if framerate != framerate_expected:
      print ("[%s] WARNING: framerate %d is not as expected %d; divide by quotient" % (strategy_name, framerate, framerate_expected))
      self.framerate_rnm = float(framerate) / framerate_expected
      #raw_input("continue? ")

    try:
      self.ddtheta_min = dicts[strategy_name]['ddtheta_min']
    except:
      print ("[%s] setting ddtheta_min=%f" % (strategy_name, self.ddtheta_min))
      pass

    #self.theta_dot_averaging = 5
    self.dtheta_averaging = dicts[strategy_name]['dtheta_averaging']
    #self.rv_delay = 50
    self.rv_delay = dicts[strategy_name]['rv_delay']
    #self.rv_averaging = 30
    self.rv_averaging = dicts[strategy_name]['rv_averaging']
    
    self.rv_min_index = dicts[strategy_name]['rv_min_index']

    self.rv_max = 1.2
    
    try:
      self.rv_const = dicts[strategy_name]['rv_const']
    except:
      self.rv_const = 0.

    # fix: if OAC settings then don't normalize          
    is_oac = self.rv_delay == 1 and self.rv_averaging == 1 and self.dtheta_averaging == 0
          
    if self.framerate_rnm != 1. and not is_oac:
      self.rv_delay = int(round(self.rv_delay * self.framerate_rnm))
      self.rv_averaging = int(round(self.rv_averaging * self.framerate_rnm))
      self.dtheta_averaging = int(round(self.dtheta_averaging * self.framerate_rnm))

      self.rv_min_index = max(0, int(round(self.rv_min_index * self.framerate_rnm)))

      print ("rv_delay = %d" % self.rv_delay)
      print ("rv_averaging = %d" % self.rv_averaging)
      print ("dtheta_averaging = %d" % self.dtheta_averaging)
      print ("rv_min_index = %d" % self.rv_min_index)

    try:
      self.use_rv_opt = dicts[strategy_name]['use_rv_opt']
    except:
      self.use_rv_opt = False

    # used for debugging only
    try:
      self._rv_opt = compute_consistent_vref_from_dicts(dicts)
      print (" [%s] (debug) consistent rv: %f" % (strategy_name, self._rv_opt))
    except Exception as e:
      if self.use_rv_opt:
        print (" [%s] unable to compute consistent rv" % (strategy_name, ))
        print (e)

    if self.use_rv_opt:
      print ("[%s] Using rv-opt. ONLY FOR DEBUGGING!" % strategy_name)

  def start(self, **kwargs):
    self.ALPHA = []
    self.theta_trace = []
    self.dtheta_trace = []
    self.dtheta_avg_trace = []
    self.rv_trace = []
    self.e_trace = []
  
  def stop(self):
    self.ALPHA = np.array(self.ALPHA)  
    self.theta_trace = np.array(self.theta_trace)
    self.dtheta_trace = np.array(self.dtheta_trace)
    self.dtheta_avg_trace = np.array(self.dtheta_avg_trace)
    self.rv_trace = np.array(self.rv_trace)
    self.e_trace = np.array(self.e_trace)

  def _average_dtheta(self, dtheta):
    if self.dtheta_averaging <= 0:
      return dtheta
      
    # averaging
    size = len(self.dtheta_trace);
    start = size - min(self.dtheta_averaging, (size - 1))
    dtheta_avg = np.mean(self.dtheta_trace[start:size])

    self.dtheta_avg_trace.append(dtheta_avg)    

    return dtheta_avg

  def _rv_index(self, size, rv_delay, rv_min=3):
    return compute_rv_index(size, rv_delay, rv_min)
  
  def _average_rv(self):
    if self.use_rv_opt:
      return self._rv_opt

    if self.rv_delay <= 0:
      return self.rv_const
    
    size = len(self.dtheta_trace)
    
    #if size <= 3:
    if size <= self.rv_min_index+1:
      return 0.

    #start = size - min(steps-1, self.rv_delay if self.rv_delay >= 3  else 2)
    #start = size - min(size-1, self.rv_delay if self.rv_delay >= 3  else 2)
    start = compute_rv_index(size, self.rv_delay, self.rv_min_index)
    end = min(start + self.rv_averaging, size-1)
    #count = max(end - start, 1)
        
    rv = np.mean( self.dtheta_trace[start:end] )

    # sanity check: OAC (comment this in if you want)
    # if self.rv_delay == 1:
    #   rv_oac = self.dtheta_trace[-2]
    #   assert(rv == rv_oac)
    #   print (" (rv == rv_oac) ", rv == rv_oac, rv, rv_oac)

    if np.abs(rv) > self.rv_max:
      rv = self.rv_max * (1 if rv>0 else -1)

    assert (not np.isnan(rv))

    return rv

  def step(self, i, x, dicts):
    alpha, _ = observe_angle_and_distance(x)
    theta = np.tan(alpha)
    self.ALPHA.append(alpha)

    u = self._step_theta(i, theta, dicts)

    #return [0., 0.] # FIXME
    return [u, 0.]


  def _step_e(self, i, e, dicts):
    if np.abs(e) <= self.ddtheta_min:
      e = 0.

    self.e_trace.append(e)
      
    u = 0.
    if e > 0.0:
      u = DynamicsModel().AGENT_A_MAX
    elif e < 0.0:
      u = -DynamicsModel().AGENT_A_MAX
      
    return u  

  def _step_dtheta(self, i, dtheta, dicts):
    rv = self._average_rv()      
    self.rv_trace.append(rv)
    
    e = (dtheta - rv)
  
    return self._step_e(i, e, dicts)

    
  def _step_theta(self, i, theta, dicts):
    self.theta_trace.append(theta)
    
    if len(self.theta_trace) < 2:
      u = 0.
    
    else:
      dtheta = (self.theta_trace[-1] - self.theta_trace[-2])/DynamicsModel().DT
      self.dtheta_trace.append(dtheta)
      
      dtheta = self._average_dtheta(dtheta)
      
      u = self._step_dtheta(i, dtheta, dicts)
    
    return u

# ----------------------------------------------------------------------------------------
class COVOAC3DStrategy(COVOAC2DStrategy):
  def __init__(self, dicts_):
    super(COVOAC3DStrategy, self).__init__(dicts_, "COVOAC3DStrategy")
    assert (DynamicsModel().dim == 3)
  
    self.BETA = []
    self.BETA_DOT = []
    self.BETA_DDOT = []

    self.BETA_DOT_FILTERED = []
    self.BETA_DDOT_FILTERED = []

    # TODO shouldn't we allow different params for IO etc.?
    self.beta_thresh = np.abs(dicts_['COVOAC3DStrategy/beta_thresh'])
    self.cba_averaging = int(np.round(np.abs(dicts_['COVOAC3DStrategy/cba_averaging'])))
    self.cba_delay = int(np.round(np.abs(dicts_['COVOAC3DStrategy/cba_delay'])))

    if self.framerate_rnm != 1.:
      self.cba_averaging = int(round(self.cba_averaging * self.framerate_rnm))
      self.cba_delay = int(round(self.cba_delay * self.framerate_rnm))

      print ("cba_averaging = %d" % self.cba_averaging)
      print ("cba_delay = %d" % self.cba_delay)

  def step(self, i, x, dicts):
    alpha, _ = observe_angle_and_distance(x)
    theta = np.tan(alpha)
    self.ALPHA.append(alpha)
    
    u = self._step_theta(i, theta, dicts)

    return self._step_cba(i, x, u, dicts)

    #return [0., 0.] # FIXME
    #return [u, 0.]
  
  
  def _step_cba(self, i,x,u,dicts):
      #global a2b_angle_init, cba_log, cba_dot_log, cba_dot_filtered_log, cba_filtered_log
      #global a2b_t0_log, a2b_t1_log

      beta_thresh = self.beta_thresh
      cba_averaging = self.cba_averaging
      cba_delay = self.cba_delay
      
      # direction towards ball
      beta = compute_bearing_angle(x)    
      beta_dot = 0.
      beta_ddot = 0.
      self.BETA.append(beta)
  
      # BETA DOT
      if len(self.BETA) > 1:
        # standard variant, no averaging
        beta_dot = (beta-self.BETA[-2]) / DynamicsModel().DT
      
        self.BETA_DOT.append(beta_dot)
      
        # beta dot averaging
        if cba_averaging > 0 and cba_delay > 0:
          beta_dot = windowed_averaged_dot(self.BETA, DynamicsModel().DT, window_size=cba_averaging, gap=cba_delay)
          self.BETA_DOT_FILTERED.append(beta_dot)
  
        # BETA DDOT
        if len(self.BETA_DOT) > 1:
          BETA_DOT_ = self.BETA_DOT
          if len(self.BETA_DOT_FILTERED) > 0:
            BETA_DOT_ = self.BETA_DOT_FILTERED 
            
          beta_ddot = (BETA_DOT_[-1]-BETA_DOT_[-2])/DynamicsModel().DT
          self.BETA_DDOT.append(beta_ddot)

          #if cba_dot_averaging > 0 and cba_dot_delay > 0:
          #  beta_ddot = windowed_averaged_dot(BETA_DOT_, DynamicsModel().DT, window_size=cba_dot_averaging, gap=cba_dot_delay)
          #  self.BETA_DDOT_FILTERED.append(beta_ddot )

      if np.isnan(beta_dot):
        raise Exception("beta_dot cannot be nan")
        
      if np.isnan(beta_ddot):
        raise Exception("beta_ddot cannot be nan")
  
      # direction towards and orthogonal to ball
      ba = np.array([x[9]-x[0], x[12]-x[6]])
      ba_orth = np.array([ba[1], -ba[0]])
      
      beta_ctrl = beta_dot
      
      u_beta = 0.
      if beta_ctrl < -beta_thresh:
        u_beta = DynamicsModel().AGENT_A_MAX
      elif beta_ctrl > beta_thresh:
        u_beta = -DynamicsModel().AGENT_A_MAX
      
      u = (u*ba)   +   (u_beta* ba_orth)
  
      # clamp
      if np.linalg.norm(u) > DynamicsModel().AGENT_A_MAX:
        u /= np.linalg.norm(u)
        u *= DynamicsModel().AGENT_A_MAX    
        
      return u


# ----------------------------------------------------------------------------------------

class AP2DStrategy(COVOAC2DStrategy):
  def __init__(self, dicts_, strategy_name='AP2DStrategy'):
    if strategy_name == 'AP2DStrategy':
      assert (DynamicsModel().dim == 2)

    super(AP2DStrategy, self).__init__(dicts_, strategy_name)
    dicts = make_hierarchical(dicts_)

    self.theta_averaging = dicts[strategy_name]['theta_averaging']
    print ("[%s] theta_averaging = %d" % (strategy_name, self.theta_averaging))

    self.framerate = dicts['BallCatching']['framerate']
    self.dt = 1./self.framerate
    
    self.ap_t = dicts[strategy_name]['ap_t']
    assert (self.ap_t in ["explicit", "implicit"])

  def step(self, i, x, dicts):
    alpha, _ = observe_angle_and_distance(x)
    theta = np.tan(alpha)
    self.ALPHA.append(alpha)

    u = self._step_theta(i, theta, dicts)

    # return [0., 0.] # FIXME
    return [u, 0.]


  def _average_theta(self, theta):
    if self.theta_averaging <= 0:
      return theta

    # averaging
    size = len(self.theta_trace);
    start = size - min(self.theta_averaging, (size - 1))
    theta_avg = np.mean(self.theta_trace[start:size])

    self.theta_avg_trace.append(theta_avg)

    return theta_avg

  def start(self, **kwargs):
    super(AP2DStrategy, self).start(**kwargs)

    self.theta_avg_trace = []
    self.t_cur = 0

    self.e_v_trace = []

  def _step_e_pd(self, i, e_p, e_v, dicts):
    #if np.abs(e) <= self.ddtheta_min:
    #  e = 0.

    self.e_trace.append(e_p)
    self.e_v_trace.append(e_v)

    #kp = 10000.
    kp = 1e3
    #kv = 1.
    kv = 0.
    return kp * e_p + kv * e_v

  def _step_theta(self, i, theta, dicts):
    self.theta_trace.append(theta)
    theta = self._average_theta(theta)

    if len(self.theta_trace) < 2:
      u = 0.

    else:
      dtheta = (self.theta_trace[-1] - self.theta_trace[-2]) / DynamicsModel().DT
      self.dtheta_trace.append(dtheta)
      dtheta = self._average_dtheta(dtheta)

      rv = self._average_rv()
      self.rv_trace.append(rv)

      t_cur = i#+1
      t_resolved = t_cur*self.dt

      # AP Mundhra
      if self.ap_t == "explicit":
        e_p = -(rv * t_resolved - theta) * self.framerate # see formulas
      # AP me
      elif self.ap_t == "implicit":
        e_p = -((self.theta_trace[-2] + rv*self.dt) - theta) * self.framerate
              
      e_v = -dtheta
      
      # FIXME
      #self.e_v_trace.append((rv * t_resolved - theta) * self.framerate)
      #self.e_v_trace.append((rv * t_resolved) - (self.theta_trace[-2] + rv*self.dt))
      # ---

#       if self.rv_delay < 0 or self.rv_delay > 1000:
#         # HACKY:  there seems to be a need for this if we use Const or IO, but not OAC
#         e_p = -e_p

      # bang-bang
      u = self._step_e(i, e_p, dicts)
      # proportional
      #u = self._step_e_pd(i, e_p, e_v, dicts)

    self.t_cur = i

    return u

