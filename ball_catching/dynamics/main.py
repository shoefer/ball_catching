# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:27:32 2016

@author: shoefer
"""

import os
import os.path
import copy
import numpy as np

from utils import *

import numpy as np

from ball_catching.dynamics.world import DynamicsModel, STATE_DIM, ACTION_DIM, \
  RecordedTrajectoryStepper, observation_noise_ball, compute_bearing_angle
from ball_catching.strategies import STRATEGIES
from ball_catching.utils.utils import *


# ----------------------------------------------------------------------------------------

def write_logs(log_folder, strategy_obj, dicts, trial, dim, X, U, DT, tf_ipol, X_noisy=None):
  # disassemble state information
  N = len(X)
  
  # t x y z vx vy vz ax ay az a_noise_x a_noise_y a_noise_z  
  agent = np.zeros( (N,13) )
  #ball = np.zeros( (N,29) )
  ball = np.zeros( (N,35) )
  ball_catching = np.zeros( (N,3) )
  
  U = np.vstack([U, np.zeros((1,U.shape[1]))])
  
  for i, x, u in zip(range(N), X, U):
    agent[i,0] = i*DynamicsModel().DT
    # position
    agent[i,1] = x[9]
    agent[i,3] = x[12]
    # velocity
    agent[i,4] = x[10]
    agent[i,6] = x[13]
    # acceleration (commanded acceleration, not actual)
    agent[i,7] = u[0] #x[11]
    agent[i,9] = u[1] #x[14]
    # 10-12 -> acceleration noise (UNUSED)

    ball[i,0] = i*DynamicsModel().DT
    # position
    ball[i,1] = x[0]
    ball[i,2] = x[3]
    ball[i,3] = x[6]
    # velocity
    ball[i,4] = x[1]
    ball[i,5] = x[4]
    ball[i,6] = x[7]
    # acceleration
    ball[i,7] = x[2]
    ball[i,8] = x[5]
    ball[i,9] = x[8]
    # 10-12 -> drag force x (UNUSED)
    # 13-15 -> noise observation 
    if X_noisy is not None:
      ball[i,13] = x[0] - X_noisy[i,0]
      ball[i,14] = x[3] - X_noisy[i,3]
      ball[i,15] = x[6] - X_noisy[i,6]
    # 16 -> noise alpha (DEPRECATED)
    # 17-19 -> wind gust force (UNUSED)
    # 20-21 -> ball projection x,y (gaze angle)
    ball[i,20] = 0. # TODO
    ball[i,21] = 0. # TODO
    # 22-23 -> noise ball projection (UNUSED)

    # 27,28 -> bearing momentum (+ noisy, unused)    
    #ball[i,27], _, _ = compute_bearing_angle(x)
    #ball[i,28], _, _ = compute_bearing_angle(X_noisy[i,:])
    # BETA
    ball[i,27] = compute_bearing_angle(x)
    # BETA noisy
    ball[i,28] = compute_bearing_angle(X_noisy[i,:])
    if i > 0:
      # BETA_DOT 
      ball[i,29] = (ball[i,27]-ball[i-1,27])/DynamicsModel().DT
      # BETA_DOT noisy
      ball[i,30] = (ball[i,28]-ball[i-1,28])/DynamicsModel().DT
      try:      
      # BETA_DOT filtered
        ball[i,33] = BETA_DOT_FILTERED[i-1]
      except:
        pass
      
      if i > 1:
        # BETA_DDOT 
        ball[i,31] = (ball[i,29]-ball[i-1,29])/DynamicsModel().DT
        # BETA_DDOT noisy
        ball[i,32] = (ball[i,30]-ball[i-1,30])/DynamicsModel().DT

        # BETA_DDOT filtered
        try:      
          ball[i,34] = BETA_DDOT_FILTERED[i-2]
        except:
          pass

    ball_catching[i,0] = i*DynamicsModel().DT # t
    # distance projected on ground
    if dim == 3:
      ball_catching[i,2] = np.sqrt( (agent[i,1]-ball[i,1])**2 + (agent[i,3]-ball[i,3])**2 )
    elif dim == 2:
      ball_catching[i, 2] = np.abs(agent[i, 1] - ball[i, 1])

  ball_catching[-1,1] = 1 # bool: ball hit ground
  
  # copy t_zero
  agent[-1,0] = tf_ipol
  ball[-1,0] = tf_ipol
  ball_catching[-1,0] = tf_ipol
  
  np.savetxt(os.path.join(log_folder, "X_%d.log" % trial), X)
  np.savetxt(os.path.join(log_folder, "U_%d.log" % trial), U)

  np.savetxt(os.path.join(log_folder, "agent_%d.log" % trial), agent)
  np.savetxt(os.path.join(log_folder, "ball_%d.log" % trial), ball)
  np.savetxt(os.path.join(log_folder, "ball_catching_%d.log" % trial), ball_catching)

  # TODO move to Strategy.write_logs
  if dicts['BallCatching/strategy'] == "COVOAC2Strategy":
    np.savetxt(os.path.join(log_folder, "rv_trace_%d.log" % trial), strategy_obj.rv_trace)

  if dicts['BallCatching/strategy'] == "COVOAC2DRawStrategy":
    np.savetxt(os.path.join(log_folder, "O_%d.log" % trial), strategy_obj._get('O'))
    np.savetxt(os.path.join(log_folder, "dO_%d.log" % trial), strategy_obj._get('dO'))
    np.savetxt(os.path.join(log_folder, "dO_dO0_%d.log" % trial), strategy_obj._get('dO_dO0'))
    np.savetxt(os.path.join(log_folder, "Z_%d.log" % trial), strategy_obj._get('Z'))


def X_as_pandas(t, X):
  import pandas as pd
  cols = ['x_b', "x_b'", "x_b''", "y_b", "y_b'", "y_b''", "z_b", "z_b'", "z_b''", "x_a", "x_a'", "x_a''", "z_a", "z_a'", "z_a''"]
  return pd.DataFrame(index = t.reshape((-1,)), data=X, columns=cols)

# pre-allocate memory to speed-up computation
x0 = np.zeros( (1, STATE_DIM) ) 
_X = np.zeros( (10000, STATE_DIM) )
_X_noisy = np.zeros( (10000, STATE_DIM) )
_U = np.zeros( (10000, ACTION_DIM) )
_t = np.zeros( (10000, 1) )


def run_python(dicts, log_root, do_write_logs=True, verbose=True):

  # Create dynamics model
  dynamics = DynamicsModel(
    framerate=float(dicts["BallCatching/framerate"]),
    r = float(dicts["Ball/radius"]),
    c = float(dicts["Ball/drag_cw"]),
    rho = float(dicts["Ball/drag_rho"]),
    mass = float(dicts["Ball/mass"]),
    a_max = float(dicts["Agent/a_max"]),
    v_max = float(dicts["Agent/v_max"]),
    sigma_agent= float(dicts["Agent/noise_std"][0]),
    sigma_ball = float(dicts["Ball/system_noise_std"][0]),
    dim = dicts['BallCatching/dim'],
    drag = dicts['Ball/drag'],
    wind_gust_force = dicts['Ball/wind_gust_force'],
    wind_gust_duration = dicts['Ball/wind_gust_duration'],
    wind_gust_relative_start_time = dicts['Ball/wind_gust_relative_start_time'],
  )
  
  trials = dicts["BallCatching/trials"]
  
  for trial in range(1,trials+1):
    print ("#####")
    print ("Trial %d" % trial)
    res = run_trial(dicts, log_root, dynamics, trial, do_write_logs=do_write_logs, verbose=verbose)
  
  # FIXME return all results, not just last (needs adaptation in caller)
  return res



def run_trial(dicts, log_root, dynamics, trial, do_write_logs=True, verbose=True):

  dim = dynamics.dim

  if verbose and dicts['Ball/drag']:
    print "Using DRAG"

  # Get strategy
  strategy_label = dicts['BallCatching/strategy']
  if strategy_label not in STRATEGIES:
    raise Exception("Strategy not supported: %s" % strategy_label)
  
  strategy_class, custom_dict = STRATEGIES[dicts['BallCatching/strategy']]
  custom_dict = make_flat({ strategy_class.__name__: custom_dict })
  dicts = copy.deepcopy(dicts)
  dicts.update(custom_dict)
  strategy_obj = strategy_class(dicts)

  #assert (dicts['Agent/x_relative'])

  # Initial agent coordinates
  if dicts['Agent/x_init_type'] == 'cartesian':
    ax_0 = dicts['Agent/x_init'][0]
    az_0 = dicts['Agent/x_init'][2]

  elif dicts['Agent/x_init_type'] == 'spherical':
    phi, _, D = dicts['Agent/x_init']
    
    ax_0 = D * np.cos(phi)
    if phi > 0:
      az_0 = np.abs(D) * np.sin(phi)
    else:
      az_0 = -np.abs(D) * np.sin(np.abs(phi))
    
  else:
    assert (False)

  delay = dicts['Agent/delay']

  
  if dicts['Ball/v_init_type'] == 'spherical':
    theta = dicts['Ball/v_init'][0]
    v_init = dicts['Ball/v_init'][2]
  
    xb_dot = v_init*np.cos(theta)
    yb_dot = np.abs(v_init)*np.sin(theta)
    zb_dot = 0.
  else:
    assert (dicts['Ball/v_init_type'] == 'cartesian')
    xb_dot = dicts['Ball/v_init'][0] 
    yb_dot = dicts['Ball/v_init'][1] 
    zb_dot = dicts['Ball/v_init'][2] 
    
    v_init = np.linalg.norm((xb_dot, yb_dot, zb_dot))
    
  if verbose:
    print ("v_init=%f, xdot = %f, ydot=%f, zb_dot=%f" % (v_init, xb_dot, yb_dot, zb_dot) )

  # if ball is thrown in the other direction we need to move
  # agent in other direction, too
  if v_init < 0:
    ax_0 = -ax_0

  # start state
  global x0

  x0[:] = ([[
  #x0 = np.array([[
          0.0, 		# 0 -> xb
          xb_dot,	# 1 -> xb' 
          0.0,    # 2 -> xb''
          0.0, 	# 3 -> yb
          yb_dot,  # 4 -> yb'
          -dynamics.GRAVITY,    # 5 -> yb''
          0.0, 	# 6 -> zb
          zb_dot,  # 7 -> zb'
          0.0,  # 8 -> zb''
          ax_0,  # 9 -> xa 
          #0.0,  # 9 -> xa # FIXME
          0.0,  # 10 -> xa'
          0.0,  # 11 -> xa''
          az_0,  # 12 -> za
          0.0,  # 13 -> za'
          0.0,  # 14 -> za''
          ]])
  
  t_n, N, x_n, z_n = dynamics.get_time_to_impact(x0)
  step_fn = dynamics.step

  #---
  # RECORDED TRAJECTORY
  rt_nm = 'Ball/recorded_trajectory'
  if rt_nm in dicts and dicts[rt_nm] is not None:
    if verbose:    
      print ("Loading recorded trajectory from %s" % dicts[rt_nm])
    step_fn = RecordedTrajectoryStepper(dicts[rt_nm])
    t_n = step_fn.tj.t.iloc[-1]#+DT # FIXME
    x_n, z_n = step_fn.tj.x.iloc[-1], step_fn.tj.z.iloc[-1]
    x0[0,:9] = step_fn.tj.iloc[0][step_fn.required_cols]
    x0 = np.nan_to_num(x0)
    #print x0
    #assert (not np.any(np.isnan(x0)))
  #---

  if verbose:
    print ( "At what time t ball hits the ground:  %f" % (t_n) )
    print ( "At what x-coordinate ball hits the ground:  %f" % (x_n) )
    print ( "At what z-coordinate ball hits the ground:  %f" % (z_n) )

  #---    
  if dynamics.is_nonlinear():
    t_n, N, x_n, z_n = dynamics.precompute_trajectory(x0)
    if verbose:
      print ( " [corrected] At what time t ball hits the ground:  %f" % (t_n) )
      print ( " [corrected] At what x-coordinate ball hits the ground:  %f" % (x_n) )


  # move agent relatively in x coordinate
  if dicts['Agent/x_relative']:
    x0[0,9] += x_n
    x0[0,12] += z_n      
  if verbose:
    print ( "Agent start position:  (%f, %f)" % (x0[0,9], x0[0,12]) )
  
  #---

  #N = int(np.ceil(t_n*DynamicsModel().FRAMERATE))  # t_n seconds at current FRAMERATE

  #if not dynamics.is_nonlinear():
  #if not dicts['Ball/drag']:
  #  N += 1
 
  # always add one more step at the end of the matrices; the interpolated
  # and the final 
  M = N+1

  # Log matrices

  # use pre-allocated data structures
  global _t, _X, _X_noisy, _U
  _t[:] = 0.
  _X[:] = 0.
  _X_noisy[:] = 0.
  _U[:] = 0.

  t = _t[:(M+1), :1]
  X = _X[:(M+1), :STATE_DIM]
  X_noisy = _X_noisy[:(M+1), :STATE_DIM]
  U = _U[:M, :ACTION_DIM]
  

  #---  
  # SIMULATION
  strategy_obj.start(dicts=dicts)
  X[0,:] = x0
  
  for i in range(0,N):    
    
    # observation noise
    if np.max(dicts['Ball/observation_noise_std']) > 0.:
      X_noisy[i,:] = observation_noise_ball(X[i,:], dicts['Ball/observation_noise_std'], dicts['Ball/noise_dist_factor'], 
              noise_dim=dim)
      # correct velocity of ball!
      # TODO correct acceleration 
      if i > 0:
        for idx in [1,4,7]:
          X_noisy[i,idx] = (X_noisy[i,idx-1]-X_noisy[i-1,idx-1])/DynamicsModel().DT
      
      #x =  X_noisy[i,:]
    else:
      X_noisy[i,:] =  X[i,:]

    if i < delay:
      # Delay is still bigger then current time step - not stepping
      #print ("DELAY - wait %d" % i)
      u = np.array([0., 0., ])

    else:
      i_rel = i - delay
      X_in = X_noisy[i_rel, :]

      #print "DELAY - STARTED %d, -> %d" % (i, i_rel)

      # Compute acceleration
      u = strategy_obj.step(i_rel, X_in, dicts)

      if strategy_obj.control_type == "acceleration":
        # Clip acceleration if too high
        assert (len(u) == 2)
        nrm_ui = np.linalg.norm(u)
        if nrm_ui > DynamicsModel().AGENT_A_MAX:
          u = DynamicsModel().AGENT_A_MAX*(np.array(u)/nrm_ui)
          #print ("u violates A_MAX -> Clipping acceleration to: " + str(u) + " (" + str(np.linalg.norm(u)) + ")")

        # Clip acceleration if resulting velocity is too high
        v_dims = [10,13]
        x_new =  step_fn(X[i,:], u, noise=False)
        v = x_new[v_dims]
        nrm_vi = np.linalg.norm(v)
        if nrm_vi > DynamicsModel().AGENT_V_MAX:
          # u = DynamicsModel().AGENT_A_MAX*(np.array(U[i])/nrm_ui)
          v_clip = DynamicsModel().AGENT_V_MAX * (np.array(v) / nrm_vi)
          v_cur = X[i, v_dims]
          u = DynamicsModel().FRAMERATE*(v_clip - v_cur)
          #print ("u violates V_MAX -> Clipping acceleration to: " + str(u) + " (" + str(np.linalg.norm(u)) + ")")

    # Advance time
    U[i] = u[:2]
    X[i+1,:] = step_fn(X[i,:], U[i,:])

    # control_type=full -> overwrite agent values by what the strategy says
    # WARN: make sure the strategy obeys the A_MAX and V_MAX limits -- they are not checked here!
    if i >= delay and strategy_obj.control_type == "full":
      assert (len(u) == 6)
      X[i+1,9:15]  = u
      # fill acceleration if ignored
      if X[i+1,11] == 0. and i >= 2:
        X[i+1,11] = (X[i-1,11]-X[i-2,11])/DynamicsModel().DT
      if X[i+1,14] == 0. and i >= 2:
        X[i+1,14] = (X[i-1,14]-X[i-2,14])/DynamicsModel().DT

      U[i,0] = X[i+1,11]
      U[i,1] = X[i+1,14]
#     print (i)
#     print (X[i+1,:])
    t[i+1] = (i+1)*DynamicsModel().DT

  # signal stop
  strategy_obj.stop()
  dynamics.set_stop()
  #---  

  # compute the interpolated distance, and add it to the end
  # of the log file; then we can compute the terminal distance correctly
  assert (X[N,3] <= 0. and X[N-1,3] >= 0.)

  t1, t2 = t[N-1:N+1]
  #ball_cols = [0,3,6]
  #agent_cols = [9,12]
  ball_cols = range(0,9)
  agent_cols = range(9, 15)
  b_prev, b = X[N-1, ball_cols], X[N, ball_cols]
  a_prev, a = X[N-1, agent_cols], X[N, agent_cols]
  #zero_offset = b_prev[1] / (b_prev[1] - b[1])
  zero_offset = b_prev[3] / (b_prev[3] - b[3])

  #print "print b, b_prev ", b, b_prev
  #print "zero_offset: ", zero_offset
  a_zero = a_prev + (a - a_prev) * zero_offset
  b_zero = b_prev + (b - b_prev) * zero_offset
  tf_ipol = t1 + zero_offset * (t2-t1)
  
  #print "a_zero  ", a_zero
  #print "b_zero  ", b_zero
  
  idist = np.linalg.norm(a_zero[[0,3]] - b_zero[[0,6]])

  if verbose:    
    print "distance(t_zero: %f) = %f" % (tf_ipol, idist)
  
  t[-1,] = tf_ipol
  X[-1, 0] = tf_ipol
  X[-1, ball_cols] = b_zero
  X[-1, agent_cols] = a_zero

  # ---

  X_noisy[N:,:] = X[N:,:]
  
    
  #---      
  if do_write_logs:
    write_parameter_file(log_root, dicts)
    write_logs(log_root, strategy_obj, dicts, trial, dim, X, U, DynamicsModel().DT, tf_ipol, X_noisy)
    strategy_obj.write_logs(log_root, trial)
  #---  

  return t, X, U, X_noisy, strategy_obj
  
  
  
#-----------------------------------------------------------------------------------------

