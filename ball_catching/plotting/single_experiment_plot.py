#!/usr/bin/python
###############################################
# Ball Catching Plotting Functions
#
# Important: Type safety; all position, orientation, velocity
# etc. values are DOUBLES and must not be written as ints

import os
import os.path
import numpy as np

import matplotlib.pylab as plt
import sys

from ball_catching.config import data_root, params_yml
from ball_catching.utils.utils import Experiment, mpl_escape_latex


###############################################
def get_tf(bc):
  T = bc.shape[0]-3 + np.argmin(bc[-3:,2]) #bc[-1,0]
  tf = bc[T,0]
  return T, tf


a, b, bc, t, T, tf  = [None]*6
def load_stats(log_root, trial):
  global a, b, bc, t, T, tf  
  
  if a is not None:
    return a, b, bc, t, T, tf  

   # load agent and ball
  a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  bc = np.loadtxt(os.path.join(log_root, "ball_catching_%d.log" % trial))
  t = a[:, 0]
  
  T, tf = get_tf(bc)
  
  if T == t.shape[0]:
    # last one - that's the interpolated value
    t = np.vstack([t[:-3], [t[-1]]])
    a = np.vstack([a[:-3], [a[-1]]])
    b = np.vstack([b[:-3], [b[-1]]])
    bc = np.vstack([bc[:-3], [bc[-1]]])
  else:
    t = t[:T+1]
    a = a[:T+1]
    b = b[:T+1]
    bc = bc[:T+1]

  return a, b, bc, t, T, tf  
###############################################
  
  
def plot_plane_positions(log_root, trial=1):
  # load agent and ball
  #a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  #b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  #t = a[:, 0]
  a, b, bc, t, T, tf  = load_stats(log_root, trial)
  
  
  plt.figure(figsize=(8,8))
  #if False and np.max(np.abs(a[:,1])) > 0 and np.max(np.abs(a[:,3])) > 0: # 
  if np.max(np.abs(a[:,1])) > 0 and np.max(np.abs(a[:,3])) > 0: # 
    plt.title("bird's eye view")
    plt.plot(a[:1,1], a[:1,3], label="agent", marker="o", color="k", ms=10.0)

    plt.plot(a[:,1], a[:,3], label="agent", color="r", lw=2.0)
    plt.plot(b[:,1], b[:,3], label="ball", color="b", lw=2.0)
    plt.xlabel("x")
    plt.ylabel("z")
    
#    plt.axis( [np.min([a[:,[1,3]], b[:,[1,3]]]), np.max([a[:,[1,3]], b[:,[1,3]]]),
#               np.min([a[:,[1,3]], b[:,[1,3]]]), np.max([a[:,[1,3]], b[:,[1,3]]]) ])
    plt.axis( [0.9*np.min(a[:,[1,3]]), 1.1*np.max([a[:,[1,3]]]),
               np.min([a[:,[1,3]], b[:,[1,5]]]), np.max([a[:,[1,3]], b[:,[1,5]]]) ])

    plt.gca().invert_yaxis()
    
    plt.legend()
    
  elif a[0,1] != b[0,1]: # agent and ball not aligned in x
    # x position
    plt.title("x position")
    plt.plot(t, np.zeros(a.shape[0]), color="k", ls="--", lw=0.5) # null line
    plt.plot(t, b[:,2], label="[ball y]", color="k", ls="--", lw=0.5)
    plt.plot(t, a[:,1], label="agent", color="r", lw=2.0)
    plt.plot(t, b[:,1], label="ball", color="b", lw=2.0)
    plt.legend(loc='lower right')
    #tm = plt.get_current_fig_manager()
    #tm.window.SetPosition((10, 10))
  
  elif a[0,3] != b[0,3]: # agent and ball not aligned in z
    # z position
    plt.title("z position")
    plt.plot(t, np.zeros(a.shape[0]), color="k", ls="--", lw=0.5) # null line
    plt.plot(t, b[:,2], label="[ball y]", color="k", ls="--", lw=0.5)
    plt.plot(t, a[:,3], label="agent", color="r", lw=2.0)
    plt.plot(t, b[:,3], label="ball", color="b", lw=2.0)
    plt.legend()
    #tm = plt.get_current_fig_manager()
    #tm.window.SetPosition((400, 10))

  else:
    print ("WARN Cannot determine along which dimension agent and ball are aligned; is x0 maybe such that agent(0)==ball(0)?" )
    print (a[0,:])
    print (b[0,:])
    return
  
  path = os.path.join(log_root, "agent_position_%d.pdf" % trial)
  plt.savefig(path, format='pdf')

  # DEPRECATED: only if we control agent rotation
  # we have bearing values
#  if a.shape[1] > 13:
#    plt.figure(figsize=(10,10))
#    plt.title("bearing angle")
#    plt.plot(t, a[:,13], color="k", ls="-", lw=2.0, label="absolute angle")
#    plt.plot(t, a[:,14], color="b", ls="-", lw=2.0, label="absolute velocity")
#    plt.plot(t, b[:,27], color="r", ls="-", lw=2.0, label="relative to ball")
#    plt.plot(t, b[:,28], color="orange", ls="-", lw=2.0, label="relative velocity")
#    path = os.path.join(log_root, "agent_bearing_%d.pdf" % trial)
#    plt.legend()
#    plt.savefig(path, format='pdf')

  ## DEPRECATED?
  # if b.shape[1] > 27:
  #   plt.figure(figsize=(5,8))
  #   ax = plt.subplot(3,1,1)
  #   ax.set_title("bearing angle")
  #   ax.plot(t, b[:,27], color="k", ls="-", lw=2.0, label="bearing angle")
  #   ax.plot(t, b[:,28], color="r", ls="--", lw=2.0, label="noisy bearing angle")
  #   #ax.set_ylim([-2,2])
  #
  #   ax = plt.subplot(3,1,2)
  #   ax.set_title("bearing momentum")
  #   ax.plot(t, b[:,29], color="k", ls="-", lw=2.0, label="bearing momentum")
  #   ax.plot(t, b[:,30], color="r", ls="--", lw=2.0, label="noisy bearing momentum")
  #   # HACKY only for python
  #   try:
  #     if np.var(b[:,33]) != 0:
  #       plt.plot(t, b[:,33], color="b", ls="-", lw=2.0, label="filtered bearing momentum")
  #   except:
  #     pass
  #
  #   b_mx, b_mn = [ fn(b[:,[29,30]]) for fn in [np.max, np.min] ]
  #   ax.set_ylim([ np.max([-5, b_mn]), np.min([5, b_mx]) ])
  #
  #   ax = plt.subplot(3,1,3)
  #   ax.set_title("bearing torque")
  #   ax.plot(t, b[:,31], color="k", ls="-", lw=2.0, label="bearing torque")
  #   ax.plot(t, b[:,32], color="r", ls="--", lw=2.0, label="noisy bearing torque")
  #   # HACKY only for python
  #   try:
  #     if np.var(b[:,34]) != 0:
  #       plt.plot(t, b[:,34], color="b", ls="-", lw=2.0, label="filtered bearing torque")
  #   except:
  #     pass
  #   b_mx, b_mn = [ fn(b[:,[31,32]]) for fn in [np.max, np.min] ]
  #   ax.set_ylim([ np.max([-5, b_mn]), np.min([5, b_mx]) ])
  #
  #   path = os.path.join(log_root, "agent_bearing_%d.pdf" % trial)
  #   #plt.legend()
  #   plt.savefig(path, format='pdf')


def plot_agent_ball_distance(log_root, framerate, trial=1):
  a, b, bc, t, T, tf  = load_stats(log_root, trial)
  
  #T = bc.shape[0]-3 + np.argmin(bc[-3:,2]) #bc[-1,0]
  #tf = bc[-1,0]
  #tf = bc[T,0]
  dist =  bc[T,2] #bc[-1,2]
  
  # agent ball distance
  plt.figure(figsize=(8,6))
  plt.title("agent ball distance")
  plt.plot(t, b[:,2], label="[ball y]", color="k", ls="--", lw=0.5)
  plt.plot(t, np.sqrt((a[:,1]-b[:,1])**2 + (a[:,3]-b[:,3])**2), label="distance", color="r", lw=2.0)
  #tm = plt.get_current_fig_manager()
  #tm.window.SetPosition((10, 400))
  
  #tf = np.argmax(bc[:,1]==1)
  #dist = np.sqrt ((a[tf,1]-b[tf,1])**2 + (a[tf,3]-b[tf,3])**2)
  
  fr = framerate
  print ("Agent-ball distance at %d (%f s): %f" % ( tf, tf/fr, dist))
  
  path = os.path.join(log_root, "agent_ball_distance_%d.pdf" % trial)
  plt.savefig(path, format='pdf')
  
  
def plot_agent_velocities(log_root, trial=1, v_max=None):
  # load agent and ball
  #a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  #b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  #t = a[:, 0]
  a, b, bc, t, T, tf  = load_stats(log_root, trial)
  
  # velocities
  plt.figure(figsize=(8,6))
  plt.title("agent velocity")
  plt.plot(t, np.zeros(a.shape[0]), color="k", ls="--", lw=0.5) # null line
  if v_max != None:
    plt.plot(t, np.ones(a.shape[0]) * v_max, color="k", ls="--", lw=0.5, label="$v_{max}$") # v_max line
    
  total = np.sqrt(a[:,4]**2+a[:,6]**2)
  plt.plot(t, b[:,2], label="[ball y]", color="k", ls="--", lw=0.5)
  #plt.plot(t, np.abs(a[:,4]), label="x", color="r", lw=1.0)
  #plt.plot(t, np.abs(a[:,6]), label="z", color="b", lw=1.0)
  plt.plot(t, a[:,4], label="x", color="r", lw=2.0)
  plt.plot(t, a[:,6], label="z", color="b", lw=2.0)
  plt.plot(t, total, label="total", ls="--", color="k", lw=1.5)
  
  plt.ylim( [1.1*np.max([-v_max, np.min(-total)]), 1.1*np.min([v_max, np.max(total)])] )
  #plt.ylim( [0., np.min([v_max, np.max(total)])] )
  
  plt.legend()
  #tm = plt.get_current_fig_manager()
  #tm.window.SetPosition((400, 400))

  #tf = a[-1,0]
  print ("Agent velocity at %f s %f" % ( tf, total[T]))

  path = os.path.join(log_root, "agent_velocity_%d.pdf" % trial)
  plt.savefig(path, format='pdf')


def plot_agent_acceleration(log_root, trial=1, a_max=None):
  # load agent and ball
  #a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  #b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  #t = a[:, 0]
  a, b, bc, t, T, tf  = load_stats(log_root, trial)
  
  plt.figure(figsize=(8,6))
  plt.title("agent acceleration")
  plt.plot(t, np.zeros(a.shape[0]), color="k", ls="--", lw=0.5) # null line
  if a_max != None:
    plt.plot(t, np.ones(a.shape[0]) * a_max, color="k", ls="--", lw=0.5, label="$a_{max}$") # a_max line
  total = np.sqrt(a[:,7]**2+a[:,9]**2)
  plt.plot(t, b[:,2], label="[ball y]", color="k", ls="--", lw=0.5)
  #plt.plot(t, np.abs(a[:,7]), label="x", color="r", lw=1.0)
  #plt.plot(t, np.abs(a[:,9]), label="z", color="b", lw=1.0)
  plt.plot(t, a[:,7], label="x", color="r", lw=2.0)
  plt.plot(t, a[:,9], label="z", color="b", lw=2.0)
  plt.plot(t, total, label="total", ls="--", color="k", lw=1.5)
  
  plt.ylim( [1.1*np.max([-a_max, np.min(-total)]), 1.1*np.min([a_max, np.max(total)])] )

  plt.legend()
  #tm = plt.get_current_fig_manager()
  #tm.window.SetPosition((400, 400))

  path = os.path.join(log_root, "agent_acceleration_%d.pdf" % trial)
  plt.savefig(path, format='pdf')

  if a.shape[1] > 10: # be backwards compatible
    if np.std(a[:,[10,12]]) > 0:
      plt.figure(figsize=(8,6))
      plt.title("agent acceleration (motor) noise")
      if np.std(a[:,[12]]) > 0:
        plt.plot(t, a[:,[12]], label="x", color="r", lw=2.0)
      if np.std(a[:,[10]]) > 0:
        plt.plot(t, a[:,[10]], label="z", color="b", lw=2.0)
      plt.legend()

      path = os.path.join(log_root, "agent_motor_noise_%d.pdf" % trial)
      plt.savefig(path, format='pdf')

def plot_agent_ball_angles(log_root, framerate, trial=1, v_max=None):
  from numpy.linalg import norm
  
  # load agent and ball
  #a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  #b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  #t = a[:, 0]
  a, b, bc, t, T, tf  = load_stats(log_root, trial)

  alpha = []
  tan_alpha = []
  dot_tan_alpha = []
  dotdot_tan_alpha = []

  for i in range(a.shape[0]):
    if a[i,0] != b[i,0]:
      raise "WARN agent and ball not aligned"
      
    a_ = a[i, 1:4]
    b_ = b[i, 1:4]
    diag = norm(b_ - a_)
    alpha.append(np.arcsin(b_[1] / diag))
    tan_alpha.append(np.tan (alpha[-1]))
    
    if len(tan_alpha) >= 2:
      dot_tan_alpha.append( (tan_alpha[-1] - tan_alpha[-2]) * framerate)
    else:
      dot_tan_alpha.append(0.)

    if len(tan_alpha) >= 3:
      dotdot_tan_alpha.append( (dot_tan_alpha[-1] - dot_tan_alpha[-2]) * framerate)
    else:
      dotdot_tan_alpha.append(0.)
  
  plt.figure(figsize=(20,20))
  
  plt.subplot(2, 2, 1)
  plt.title("tan alpha (actual)")
  plt.plot(t, tan_alpha, lw=2.0)
  plt.ylim( [-0.1, 2.0] )

  plt.subplot(2, 2, 2)
  plt.title("tan alpha dot (actual)")
  plt.plot(t, dot_tan_alpha, lw=2.0)
  plt.ylim( [-2.0, 2.0] )

  plt.subplot(2, 2, 3)
  plt.title("tan alpha dotdot (actual)")
  plt.plot(t, dotdot_tan_alpha, lw=2.0)
  plt.ylim( [-2.0, 2.0] )

  plt.subplot(2, 2, 4)
  plt.title("alpha (actual)")
  plt.plot(t, alpha, lw=2.0)
  plt.ylim( [-2.0, 2.0] )  

  path = os.path.join(log_root, "agent_ball_angles_%d.pdf" % trial)
  plt.savefig(path, format='pdf')
  
  
def compute_rv(params, a):
  if params['Ball']['v_init_type'] == "spherical":
    V = params['Ball']['v_init'][2]
    theta = params['Ball']['v_init'][0]
    a0 = a[0,1]
    
    print "V=%f" % V
    print "theta=%f" % theta
    print "sin(theta)=%f" % np.sin(theta)
    print "a0=%f" % a0
    
    rv = np.sin(theta)*V/a0
    print "rv=%f" % rv
    
    return rv
    
  else:
    print "[plot_oac] Cannot compute rv*; Ball.v_init_type!=spherical"
    
    return None
  
def compute_daref(params, a):
  if params['Ball']['v_init_type'] == "spherical":
    V = params['Ball']['v_init'][2]
    theta = params['Ball']['v_init'][0]
    a0 = a[0,1]
    g = 9.81
    
    print ("----")
    print "V=%f" % V
    print "theta=%f" % theta
    print "sin(theta)=%f" % np.sin(theta)
    print "a0=%f" % a0
    
    daref =  V*np.cos(theta) - (a0*g)/(2*V*np.sin(theta))
    print "daref=%f" % daref
    
    return daref
    
  else:
    print "[plot_oac] Cannot compute rv*; Ball.v_init_type!=spherical"
    
    return None  
      
def plot_oac(params, log_root, type="OACStrategy", trial=1):
  #a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  #b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  #t = a[:, 0]
  a, b, bc, t, T, tf  = load_stats(log_root, trial)
  
  oac = np.loadtxt(os.path.join(log_root, "%s_%d.log" % (type, trial)))
  
  plt.figure(figsize=(20,15))
  plt.subplot(2, 2, 1)
  plt.title("tan alpha (sensed)")
  plt.plot(t, oac[:, 4], lw=2.0)
  plt.ylim( [-0.1, 2.0] )

  plt.subplot(2, 2, 2)
  plt.title("tan alpha dot (sensed)")
  plt.plot(t, oac[:, 5], lw=2.0, label=r'$\frac{d}{dt} \tan \alpha$')
  # get optimal rv
  rv = compute_rv(params, a)
  plt.plot(t, np.ones( (t.shape[0], )) * rv, ls="--", lw=2.0, label="rv*")
  plt.legend()
  plt.ylim( [-2.0, 2.0] )

  plt.subplot(2, 2, 3)
  plt.title("tan alpha dotdot (sensed)")
  plt.plot(t, oac[:, 6], lw=2.0)
  plt.plot(t, [0.03]*t.shape[0], "--", lw=2.0)
  plt.plot(t, [-0.03]*t.shape[0], "--", lw=2.0)
  plt.ylim( [-2.0, 2.0] )

  plt.subplot(2, 2, 4)
  plt.title("alpha (sensed)")
  plt.plot(t, oac[:, 7], lw=2.0)
  plt.ylim( [-2.0, 2.0] )

  path = os.path.join(log_root, "oac_alpha_%d.pdf" % trial)
  plt.savefig(path, format='pdf')

  if type == "OACStrategy":
    plt.figure()
    plt.plot(t, oac[:, 9], lw=2.0, label="xdotdot")
    plt.plot(t, oac[:, 8], lw=2.0, label="vdotdot")
    plt.legend()
    #plt.ylim( [-2.0, 2.0] )

    path = os.path.join(log_root, "oac_Fdes_%d.pdf" % trial)
    plt.savefig(path, format='pdf')

  elif type == "OACConstAccStrategy":
    plt.figure()
    plt.plot(t, oac[:, 0], lw=2.0, label="v_des")
    plt.legend()

    path = os.path.join(log_root, "oac_vdes_%d.pdf" % trial)
    plt.savefig(path, format='pdf')

  plt.figure()
  plt.plot(t, b[:, 13], lw=2.0, label="angle noise (OAC)")
  plt.legend()

  path = os.path.join(log_root, "oac_alpha_noise_%d.pdf" % trial)
  plt.savefig(path, format='pdf')

def plot_cov(params, log_root, type="COVStrategy", trial=1):
  #a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  #b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  #t = a[:, 0]
  a, b, bc, t, T, tf  = load_stats(log_root, trial)
  
  oac = np.loadtxt(os.path.join(log_root, "%s_%d.log" % (type, trial)))

  if False: # FIXME
      plt.figure(figsize=(8,6))
      plt.subplot(3, 1, 1)
      plt.title("tan alpha (sensed)")
      plt.plot(t, oac[:, 4], lw=2.0)
      plt.ylim( [-0.1, 2.0] )
    
      plt.subplot(3, 1, 2)
      plt.title("tan alpha dot (sensed)")
      plt.plot(t, oac[:, 5], lw=2.0)
      # get optimal rv
      rv = compute_rv(params, a)
      plt.plot(t, np.ones( (t.shape[0], )) * rv, ls="--", lw=2.0, label="rv*")
      plt.legend()
      plt.ylim( [-2.0, 2.0] )
    
      plt.subplot(3, 1, 3)
      plt.title("alpha (sensed)")
      plt.plot(t, oac[:, 7], lw=2.0)
      plt.ylim( [-3.2, 3.2] )
    
      path = os.path.join(log_root, "cov_alpha_%d.pdf" % trial)
      plt.savefig(path, format='pdf')
    
      plt.figure()
      plt.plot(t, oac[:, 0], lw=2.0, label="v_des")
      plt.legend()
    
      path = os.path.join(log_root, "cov_vdes_%d.pdf" % trial)
      plt.savefig(path, format='pdf')
    
      plt.figure()
      plt.title("angle noise (OAC)")
      plt.plot(t, b[:, 13], lw=2.0)
      plt.legend()
      path = os.path.join(log_root, "cov_alpha_noise_%d.pdf" % trial)
      plt.savefig(path, format='pdf')
    
      plt.figure()
      plt.title("COV error")
      plt.plot(t, oac[:, 9], lw=2.0)
      path = os.path.join(log_root, "cov_error_%d.pdf" % trial)
      plt.savefig(path, format='pdf')
    
      plt.figure()
      plt.title("rv")
      plt.plot(t, oac[:, 10], lw=2.0)
      path = os.path.join(log_root, "rv_%d.pdf" % trial)
      plt.savefig(path, format='pdf')

  try:
    plt.figure()
    plt.title("COV: bearing")
    #plt.plot(t, oac[:, 11], lw=2.0, label="error p")
    plt.plot(t, oac[:, 11], lw=2.0, label="bearing momentum")
    plt.plot(t, oac[:, 12], lw=2.0, label="bearing torque")
  except:
    print ("cannot plot COV bearing - old log files?")
    
  #plt.plot(t, oac[:, 13], lw=2.0, label="error i")
  plt.legend(loc="lower left")
  path = os.path.join(log_root, "cov_bearing_%d.pdf" % trial)
  plt.savefig(path, format='pdf')


def plot_tp(log_root, type="OACStrategy", trial=1):
  #a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  #b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  #t = a[:, 0]
  a, b, bc, t, T, tf  = load_stats(log_root, trial)

  tp = np.loadtxt(os.path.join(log_root, "TrajectoryPredictionstrategy_%d.log" % trial))
  
  plt.figure( (10, 15) )

  plt.subplot(2, 1, 1)
  plt.title("ball landing point: prediction vs. actual")
  plt.plot(t, tp[:, 1], lw=2.5, c='b', label="x_predicted")
  plt.plot(t, a[:, 1], lw=1.5, c='g', label="x agent")
  plt.plot(t, tp[:, 5], lw=1.5, c='c', label="dotdot(x) agent")
  plt.plot(t, [b[-1, 1]] * t.shape[0], lw=1.0, ls="--", c='k', label="x")
  plt.legend()

  plt.subplot(2, 1, 2)
  plt.title("ball landing point: prediction vs. actual")
  plt.plot(t, tp[:, 3], lw=2.5, c='b', label="z_predicted")
  plt.plot(t, a[:, 3], lw=1.5, c='g', label="z agent")
  plt.plot(t, tp[:, 7], lw=1.5, c='c', label="dotdot(z) agent")
  plt.plot(t, [b[-1, 3]] * t.shape[0], lw=1.0, ls="--", c='k', label="z")
  #plt.ylim( [-0.1, 2.0] )
  plt.legend()

  path = os.path.join(log_root, "tp_%d.pdf" % trial)
  plt.savefig(path, format='pdf')

  
def plot_ball_trajectory(log_root, type="OACStrategy", trial=1):
  #b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  a, b, bc, t, T, tf  = load_stats(log_root, trial)
  
  plt.figure(figsize=(10,15))
  plt.subplot(2, 1, 1)
  plt.title("ball x/y")
  plt.plot(b[:, 1], b[:, 2], lw=2.0)
  plt.xlabel("x")
  plt.ylabel("y")

  plt.subplot(2, 1, 2)
  plt.title("ball x/z")
  plt.plot(b[:, 1], b[:, 2], lw=2.0)
  plt.xlabel("x")
  plt.ylabel("y")

  path = os.path.join(log_root, "ball_trajectory_%d.pdf" % trial)
  plt.savefig(path, format='pdf')

  if np.max(b[:,13:16]) > 0:
    plt.figure(figsize=(8,8))
    plt.subplot(1, 1, 1)
    plt.title("ball observation noise x/y/z")
    plt.plot(b[:, 0], b[:, 13], lw=2.0, label="x")
    plt.plot(b[:, 0], b[:, 14], lw=2.0, label="y")
    plt.plot(b[:, 0], b[:, 15], lw=2.0, label="z")
    plt.xlabel("t")
    plt.legend()

    path = os.path.join(log_root, "ball_observation_noise_%d.pdf" % trial)
    plt.savefig(path, format='pdf')

  if np.max(b[:,24:27]) > 1e-15:
    plt.figure(figsize=(8,12))
    #plt.subplot(2, 1, 1)
    plt.title("ball system noise x/y/z")
    plt.plot(b[:, 0], b[:, 24], lw=2.0, label="x")
    plt.plot(b[:, 0], b[:, 25], lw=2.0, label="y")
    plt.plot(b[:, 0], b[:, 26], lw=2.0, label="z")
    plt.xlabel("t")

    path = os.path.join(log_root, "ball_system_noise_%d.pdf" % trial)
    plt.savefig(path, format='pdf')


def plot_single_experiment(log_root, dictionaries, experiment_sets, experiment_root):
  if len(experiment_sets) > 1 or len(experiment_sets.values()[0]) > 1:
    print "WARN: Cannot plot a single experiment because experiment set contains several experiments"

    # create a symlink "last" in multi-experiment folder
    symlink = os.path.join(data_root, "last")
    if os.path.islink(symlink):
      os.unlink(symlink)
    if os.path.exists(symlink):
      print ("WARN: last -> %s last cannot be set as symlink because it is a file or folder!" % experiment_root)
    else:
      os.symlink(experiment_root, symlink)

  else:

    # create a symlink "last" in single experiment folder
    symlink = os.path.join(data_root, "last")
    if os.path.islink(symlink):
      os.unlink(symlink)
    if os.path.exists(symlink):
      print ("WARN: last -> %s last cannot be set as symlink because it is a file or folder!" % log_root)
    else:
      os.symlink(log_root, symlink)

    plot_plane_positions(log_root)
    # plot_agent_ball_distance(log_root, dictionaries['BallCatching']['framerate'])
    plot_agent_velocities(log_root, v_max=dictionaries['Agent']['v_max'] if 'v_max' in dictionaries['Agent'] else None)
    plot_agent_acceleration(log_root,
                            a_max=dictionaries['Agent']['a_max'] if 'a_max' in dictionaries['Agent'] else None)
    # plot_ball_trajectory(log_root)

    # if dictionaries['BallCatching']['strategy'] == 'OACStrategy' or  dictionaries['BallCatching']['strategy'] == 'OACConstAccStrategy':
    # plot_oac(dictionaries, log_root, dictionaries['BallCatching']['strategy'])
    if dictionaries['BallCatching']['strategy'] == 'COVStrategy':
      plot_cov(dictionaries, log_root, dictionaries['BallCatching']['strategy'])

      # else:
      # plot_agent_ball_angles(log_root, dictionaries['BallCatching']['framerate'])

      # if dictionaries['BallCatching']['strategy'] == 'TrajectoryPredictionStrategy':
      # plot_tp(log_root)

  plt.show()


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print ("Usage: single_experiment_plot.py <experiment_name> [<trial>=1]")

  e = Experiment(os.path.join(data_root, sys.argv[1]))
  p = e.get_params()
  
  trial = 1
  if len(sys.argv) >=3 :
    trial = int(sys.argv[2])
  print "Plotting trial %d" % trial

  plot_plane_positions(e.get_log_root(), trial=trial)
  
  plot_agent_ball_distance(e.get_log_root(), p['BallCatching']['framerate'], trial=trial)
  plot_agent_velocities(e.get_log_root(), v_max=p['Agent']['v_max'] if 'v_max' in p['Agent'] else None, trial=trial )
  plot_agent_acceleration(e.get_log_root(), a_max=p['Agent']['a_max'] if 'a_max' in p['Agent'] else None, trial=trial )

  plot_ball_trajectory(e.get_log_root(), trial=trial)
  
  #if p['BallCatching']['strategy'] == 'OACStrategy' or  p['BallCatching']['strategy'] == 'OACConstAccStrategy':
    #plot_oac(p, e.get_log_root(), p['BallCatching']['strategy'], trial=trial)
  if p['BallCatching']['strategy'] == 'COVStrategy' or p['BallCatching']['strategy'] == 'APStrategy':
    print ("plotting OAC and COV")
    plot_oac(p, e.get_log_root(), p['BallCatching']['strategy'], trial=trial)
    #plot_cov(p, e.get_log_root(), p['BallCatching']['strategy'], trial=trial)
  #else:
    #plot_agent_ball_angles(e.get_log_root(), p['BallCatching']['framerate'], trial=trial)

  #if p['BallCatching']['strategy'] == 'TrajectoryPredictionStrategy':
    #plot_tp(e.get_log_root(), trial=trial)

  plt.show()