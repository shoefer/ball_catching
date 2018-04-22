#!/usr/bin/python
###############################################
# Ball Catching Plotting Functions
#
# Important: Type safety; all position, orientation, velocity
# etc. values are DOUBLES and must not be written as ints

from comparison_plot import pretty_strategy_names_py as pretty_strategy_names

import os
import os.path
import numpy as np
import argparse
import matplotlib
#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = [\
    #r"\usepackage{accents}",]
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rcParams['font.size'] = 30

#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman'], 'size': 30})

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

import sys

from src.ball_catching.config import data_root, params_yml
from src.ball_catching.utils import Experiment, pdfcrop

from collections import OrderedDict

from numpy.linalg import norm
from single_experiment_plot import compute_rv, compute_daref, get_tf

from slugify import slugify

#pretty_strategy_names = OrderedDict( (
#  ("KalmanLQRStrategy", "LQG"),
#  ("LQRStrategy", "LQR only"), 
#  ("KalmanLQRV10Strategy", "LQG (vel=0.01)"), 
#  ("LQRV10Strategy", "LQR only (vel=0.01)"), 
#  ("KalmanLQRV1000Strategy", "LQG (vel=1)"), 
#  ("LQRV1000Strategy","LQR only (vel=1)"), 
#
#  ("ELQRStrategy", "iLQR"), 
#  ("ELQRSoccerStrategy", "iLQR (soccer)"),
#  
#  ("OACStrategy", "OAC"),
#  ("JerkOACStrategy", "OAC-jerk"), 
#  ("ConstCOVStrategy", "COV"), 
#  ("ObsInitCOVStrategy", "COV-IO"),
#  ("OACCOVStrategy", "COV-OAC"),
#) ) 

extension = "pdf"

################################################################

def load_log_files(log_root, strategy, trial=1):
  a = np.loadtxt(os.path.join(log_root, "agent_%d.log" % trial))
  b = np.loadtxt(os.path.join(log_root, "ball_%d.log" % trial))
  bc = np.loadtxt(os.path.join(log_root, "ball_catching_%d.log" % trial))

  # strategy
  try:
    s = np.loadtxt(os.path.join(log_root, "%s_%d.log" % (strategy, trial)))
  except:
    try:
        s = np.loadtxt(os.path.join(log_root, "%s_%d.log" % (strategy.lower(), trial)))
    except:
        s = None
  
  return a,b,bc,s 


def get_pretty_strategy_name(e):
  raw_name = ""
  name = None
  for k,v in pretty_strategy_names.items():
    if k in e.get_log_root():
      if len(k) > len(raw_name):
        raw_name = k
        name = v

  if name is None:
    print ("Warn: strategy not found")
    return "???"

  return name

def generate_paper_title(e, pname):
  strategy = slugify(get_pretty_strategy_name(e))
  V = e.get_params()['Ball']['v_init'][2]
  D = np.sqrt( e.get_params()['Agent']['x_init'][0]**2 + e.get_params()['Agent']['x_init'][2]**2)
  D_str = ("D%.2f" % D)
  if e.get_params()['Agent']['x_init_type'] == "spherical":
      D = e.get_params()['Agent']['x_init'][0]
      phi = e.get_params()['Agent']['x_init'][2]
      D_str = ("D%.2f_phi_%.2f" % (D, phi) )
  
  pn = slugify(pname)
  return os.path.join(e.get_log_root(), strategy + "_" + ("V%.2f" % V) + "_" + D_str + "_" + pn + "." + extension)
  

# --------------------------------------------------------------

def get_agent_ball_positions(params, a, b, bc=None, strategy=None):
  """
    t, ax, az, bx, bz
  """
  t = a[:, 0]
  return t, a[:,1], a[:,3], b[:,1], b[:,3]

def get_agent_ball_distance(params, a, b, bc=None, strategy=None):
  t = a[:, 0]
  return t, np.sqrt((a[:,1]-b[:,1])**2 + (a[:,3]-b[:,3])**2)

def get_absolute_agent_velocity(params, a, b, bc=None, strategy=None):
  t = a[:, 0]
  total = np.sqrt(a[:,4]**2+a[:,6]**2)
  return t, total,a[:,4],a[:,6]

def get_absolute_agent_acceleration(params, a, b, bc=None, strategy=None):
  t = a[:, 0]
  total = np.sqrt(a[:,7]**2+a[:,9]**2)
  return t, total,a[:,7],a[:,9]

def get_angle_statistics(params, a, b, bc=None, strategy=None):
  t = a[:, 0]
  
  framerate = params['BallCatching']['framerate']
  
  alpha = []
  tan_alpha = []
  dot_tan_alpha = []
  dotdot_tan_alpha = []

  for i in range(a.shape[0]):
    if a[i,0] != b[i,0]:
      raise "WARN agent and ball not aligned"
      
    a_ = a[i, 1:4]
    b_ = b[i, 1:4]
    
    # add observation noise to show how it's seen by agent
    b_ += b[i, 13:16]
    
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

  return t, tan_alpha, dot_tan_alpha, dotdot_tan_alpha
  
# --------------------------------------------------------------
  
def plot_position(e,stats,force_dim=None, title=None, filename_append=None):
  t, ax, az, bx, bz = get_agent_ball_positions(e.get_params(), *stats)
  T, tf = get_tf(stats[2])
  
  fig = plt.figure(figsize=(8,8))
  padding = 0.1

  #print "force_dim ", force_dim
  
  #ax = fig.add_subplot(1,1,1)  

  # 3D case
  if np.max(ax) > 0 and np.max(az) > 0 and (force_dim is None and force_dim != 2):
    if title is None:
      plt.title("bird's eye view")
    else:
      plt.title(title)
    
    # only evaluate till T  
    ax, az, bx, bz = ax[:T+1], az[:T+1], bx[:T+1], bz[:T+1]
      
    #plt.scatter(ax[:,:1], az[:,:1], color="r", ms=50, )
    plt.plot(ax, az, label="agent", color="r", lw=5.0)
    #plt.scatter(bx[:,:1], bz[:,:1], color="b", ms=50, )
    plt.plot(bx, bz, label="ball", color="b", lw=5.0)

    plt.xlabel("x")
    plt.ylabel("z")
    
    x_mn, x_mx = np.min([np.min(ax), np.max(bx)]), np.max(ax)
    z_mn, z_mx = np.min(az), np.max(az)
    
    print x_mn, x_mx    
    
    axis = plt.axes()

    # agent arrows
    axis.arrow(ax[len(ax)/3], az[len(az)/3],
               ax[len(ax)/3+1]-ax[len(ax)/3], az[len(az)/3+1]-az[len(az)/3], 
               head_width=1.5, head_length=1.5, fc='r', ec='r')
    axis.arrow(ax[len(ax)/1.5], az[len(az)/1.5],
               ax[len(ax)/1.5+1]-ax[len(ax)/1.5], az[len(az)/1.5+1]-az[len(az)/1.5], 
               head_width=1.5, head_length=1.5, fc='r', ec='r')

    #ball arrow
    pos = -10
    axis.arrow(bx[len(bx)+pos], bz[len(bz)+pos],
               bx[len(ax)+pos+1]-bx[len(ax)+pos], bz[len(az)+pos+1]-bz[len(az)+pos], 
               head_width=1.5, head_length=1.5, fc='b', ec='b')

    if x_mx - x_mn > z_mx-z_mn:
      z_mx += (x_mx - x_mn)/2.
      z_mn -= (x_mx - x_mn)/2.
    else:
      x_mx += (z_mx - z_mn)/2.
      x_mn -= (z_mx - z_mn)/2.
    
    pad_xmin = 1.
    if np.min(ax) > np.max(bx) - 1:
      pad_xmin = 3
    plt.axis( [x_mn - pad_xmin, x_mx + 1, z_mn - 1., z_mx + 1. ])
    plt.gca().invert_yaxis()
    
    #a = ax.tolist() + az.tolist()
    #b = bx.tolist() + bz.tolist()

    leg = plt.legend(loc='lower left', ) #fancybox=True, )
    #plt.legend(loc='upper right')
    leg.get_frame().set_alpha(0.5)
      
    
  # 2D case
  else:
    axis = fig.add_subplot(1,1,1)
    plt.title("Agent/ball position")
    plt.plot(t, ax, label="agent", color="r", lw=5.0)
    plt.plot(t, bx, label="ball", color="b", lw=5.0)

    # FIXME does not work
    # # agent arrows
    # off1, off2 = 3, 10
    # axis.arrow(t[len(ax)/3], ax[len(ax)/3],
    #            t[len(ax)/3+off1]-t[len(ax)/3], ax[len(ax)/3+off2]-ax[len(ax)/3],
    #            head_width=1.5, head_length=1.5, fc='r', ec='r')
    # axis.arrow(t[len(ax)/1.5], ax[len(ax)/1.5],
    #            t[len(ax)/1.5+1]-t[len(ax)/1.5], ax[len(ax)/1.5+1]-ax[len(ax)/1.5],
    #            head_width=1.5, head_length=1.5, fc='r', ec='r')
    #
    # #ball arrow
    # pos = -10
    # off1, off2 = 1,2
    # axis.arrow(t[len(bx)+pos], bx[len(bx)+pos],
    #            t[len(bx)+pos+off1]-t[len(bx)+pos], bx[len(bx)+pos+off2]-bx[len(bx)+pos],
    #            head_width=1.5, head_length=1.5, fc='b', ec='b')


    plt.scatter( t[:1], bx[:1], color="b", s=200, )
    plt.scatter( t[-1:], bx[-1:], color="w", s=200, )
    plt.scatter( t[:1], ax[:1], color="r", s=200, )
    plt.scatter( t[-1:], ax[-1:], color="k", s=200, )

    plt.xlabel("t")
    plt.ylabel("x")
    
    a = ax.tolist()
    b = bx.tolist()
    
    x_len = np.max(t)-np.min(t)
    y_len = np.max(a+b)-np.min(a+b)
    
    plt.xlim(np.min(t)-padding*x_len, np.max(t)+padding*x_len)
    plt.ylim(np.min(a+b)-padding*y_len, np.max(a+b)+padding*y_len)
    
    leg = plt.legend(loc='lower right', fontsize=24, )# fancybox=True, )
    leg.get_frame().set_alpha(0.5)
  
  fn = generate_paper_title(e, "agent_ball_distance"+filename_append)
  print ("saving "+fn)
  plt.savefig(fn)
  pdfcrop(fn)
  
  
def plot_agent_acceleration(e,stats, force_dim = None, title=None, filename_append=None):
  t, total, x, z = get_absolute_agent_acceleration(e.get_params(), *stats)

  plt.figure(figsize=(8,8))
  padding = 0.1

  plt.title("Agent acceleration")

  plt.xlabel("t")
  if np.max(z) != 0 and (force_dim is None and force_dim != 2): # 3D
    plt.ylabel(r"absolute acceleration $\sqrt {\ddot{a}_x^2 + \ddot{a}_z^2}$")
    plt.plot(t, total, label="agent", color="r", lw=5.0)
  else: # 2D
    plt.ylabel(r"$\ddot{a}_x$")
    plt.plot(t, x, label="agent", color="r", lw=5.0)

  a_max = e.get_params()['Agent']['a_max']
  if a_max is not None:
    plt.plot(t, len(t)*[-a_max], label="minimum acceleration", ls="--", color="k", lw=3.0)
    plt.plot(t, len(t)*[a_max], label="maximum acceleration", ls="--", color="k", lw=3.0)
    
    if np.max(z) != 0 and (force_dim is None and force_dim != 2): # 3D; absolute value
      plt.ylim(0, a_max*1.1)
    else: # 2D; +/- values
      plt.ylim(-(1+padding)*a_max, (1+padding)*a_max)
      plt.plot(t, len(t)*[0], ls="-", color="k", lw=1.0)
  
  #x_len = np.max(t)-np.min(t)
  #y_len = np.max(a+b)-np.min(a+b)
  
  #plt.xlim(np.min(t)-padding*x_len, np.max(t)+padding*x_len)
  #plt.ylim(np.min(a+b)-padding*y_len, np.max(a+b)+padding*y_len)
  
  #plt.legend(loc='lower right')
  fn = generate_paper_title(e, "agent_acceleration"+filename_append)
  print ("saving "+fn)
  plt.savefig(fn)

def plot_agent_velocity(e,stats, force_dim=None, title=None, filename_append=None):
  t, total, x, z = get_absolute_agent_velocity(e.get_params(), *stats)
  
  plt.figure(figsize=(8,8))
  
  padding = 0.1

  plt.title("Agent velocity")

  plt.xlabel("$t$")
  if np.max(z) != 0 and (force_dim is None and force_dim != 2): # 3D
    plt.ylabel(r"absolute velocity $\sqrt {\dot{a}_x^2 + \dot{a}_z^2}$")
    plt.plot(t, total, label="agent", color="r", lw=5.0)
  else: # 2D
    plt.ylabel(r"$\dot{a}_x$")
    plt.plot(t, x, label="agent", color="r", lw=5.0)

  v_max = e.get_params()['Agent']['v_max']
  if v_max is not None:
    plt.plot(t, len(t)*[-v_max], label="minimum velocity", ls="--", color="k", lw=3.0)
    plt.plot(t, len(t)*[v_max], label="maximum velocity", ls="--", color="k", lw=3.0)
    
    if np.max(z) != 0 and (force_dim is None and force_dim != 2): # 3D; absolute value
      plt.ylim(0, v_max*1.1)
    else: # 2D; +/- values
      plt.ylim(-(1+padding)*v_max, (1+padding)*v_max)
      plt.plot(t, len(t)*[0], ls="-", color="k", lw=1.0)

  
  #x_len = np.max(t)-np.min(t)
  #y_len = np.max(a+b)-np.min(a+b)
  
  #plt.xlim(np.min(t)-padding*x_len, np.max(t)+padding*x_len)
  #plt.ylim(np.min(a+b)-padding*y_len, np.max(a+b)+padding*y_len)
  
  #plt.legend(loc='lower right')
  fn = generate_paper_title(e, "agent_velocity"+filename_append )
  print ("Saving "+fn)
  plt.savefig(fn)

def plot_viewing_angle(e,stats,title=None, filename_append=None):
  t, tan_alpha, dot_tan_alpha, dotdot_tan_alpha = get_angle_statistics(e.get_params(), *stats)
  
  fig = plt.figure(figsize=(10,10))
  #padding = 0.1

  plt.title("Tangent of the vertical viewing angle $\theta$")

  plt.xlabel("t")
  #plt.ylabel(r"")

  rv = compute_rv(e.get_params(), stats[0])

  ta_title = r"$\theta$"
  dta_title = r"$\dot{\theta}$"
  ddta_title = r"$\ddot{\theta}$"

  gs1 = gridspec.GridSpec(3,1)
  gs1.update(hspace=0.6) # set the spacing between axes.
  
  plt.subplot(gs1[0])
  plt.title(ta_title)
  plt.plot(t, tan_alpha, label=None, color="g", lw=5.0)
  plt.ylim([0, 2.5])

  plt.subplot(gs1[1])
  plt.title(dta_title)
  plt.plot(t, dot_tan_alpha, label=None, color="b", lw=5.0)
  plt.plot(t, len(t)*[rv], label=r"$\dot{theta}^*_\mathrm{ref}$", ls="--", color="k", lw=3.0)
  plt.plot(t, len(t)*[0], ls="-", color="k", lw=1.0)
  plt.ylim([0, 0.5])

  # FIXME!!!!!!!!!!!!!!
  # comment this in for creating the adversarially chosen reference velocity plot
  #rv_dev = -0.02
  #plt.plot(t, len(t)*[rv+rv_dev], label=r"$\dot{theta}_\mathrm{ref}$", ls="--", color="gray", lw=3.0)
  #plt.ylim([rv+rv_dev-0.1, rv+0.1])
  
  plt.subplot(gs1[2])
  plt.title(ddta_title)
  plt.plot(t, dotdot_tan_alpha, label=ddta_title, color="r", lw=5.0)
  plt.plot(t, len(t)*[0], ls="-", color="k", lw=1.0)
  plt.ylim([-15, 15.])

  # FIXME!!!!!!!!!!!!!!
  # comment this in for creating the adversarially chosen reference velocity plot
  #plt.ylim([-1, 1.])

  #plt.tight_layout()
 
  #plt.legend()
  fn = generate_paper_title(e, "vva"+filename_append)
  print ("Saving "+fn)
  plt.savefig(fn)
  

# --------------------------------------------------------------

def plot_velocity_agent_and_vva(e,stats,force_dim,title, filename_append):
  t, tan_alpha, dot_tan_alpha, dotdot_tan_alpha = get_angle_statistics(e.get_params(), *stats)
  t, total, x, z = get_absolute_agent_velocity(e.get_params(), *stats)

  matplotlib.rcParams['font.size'] = 26
  
  padding = 0.1

  rv = compute_rv(e.get_params(), stats[0])
  daref = compute_daref(e.get_params(), stats[0])

  fig = plt.figure(figsize=(9,10))

  gs1 = gridspec.GridSpec(2,1)
  gs1.update(hspace=0.3) # set the spacing between axes.

  # theta dot
  plt.subplot(gs1[0])
  dta_title = r"$\dot{\theta}$"
  plt.title(dta_title)
  plt.plot(t, dot_tan_alpha, label=dta_title, color="b", lw=5.0)
  plt.plot(t, len(t)*[rv], label=r"$\dot{\theta}^*_\mathrm{ref}$", ls="--", color="gray", lw=3.0)
  plt.plot(t, len(t)*[0], ls="-", color="k", lw=1.0)
  plt.ylim([0, 0.5])
  plt.xlabel("$t$")
  plt.xlim([t[0], t[-1]])
  plt.legend(loc="upper left", ncol=2)
  
  # aref dot
  daref_title = r"$\dot{a}^*_\mathrm{ref}$"
  
  plt.subplot(gs1[1])
  plt.xlabel("$t$")
  
  if np.max(z) != 0 and (force_dim is None and force_dim != 2): # 3D
    plt.ylabel(r"absolute velocity $\sqrt {\dot{a}_x^2 + \dot{a}_z^2}$")
    plt.plot(t, total, label="agent", color="r", lw=5.0)
  else: # 2D
    plt.ylabel(r"$\frac{m}{s}$")
    plt.plot(t, x, label="$\dot{a}_x$", color="r", lw=5.0)

  plt.plot(t, len(t)*[daref], label=daref_title, ls="--", color="gray", lw=3.0)

  v_max = e.get_params()['Agent']['v_max']
  if v_max is not None:
    plt.plot(t, len(t)*[-v_max], ls="--", color="k", lw=3.0, label=r"$\pm a_\mathrm{max}$", )
    plt.plot(t, len(t)*[v_max], ls="--", color="k", lw=3.0, ) #label="maximum velocity", ) 
    
    if np.max(z) != 0 and (force_dim is None and force_dim != 2): # 3D; absolute value
      plt.ylim(0, v_max*1.1)
    else: # 2D; +/- values
      plt.ylim(-(1+padding)*v_max, (1+padding)*v_max)
      plt.plot(t, len(t)*[0], ls="-", color="k", lw=1.0)  

  plt.xlim([t[0], t[-1]])
  plt.legend(loc="upper left", ncol=2)

  fn = generate_paper_title(e, "daref_vva"+filename_append)
  print ("Saving "+fn)
  plt.savefig(fn)

# --------------------------------------------------------------

def plot(e, trial=1, force_dim=None, title=None, size="m", filename_append=""):
  p = e.get_params()
  
  if size == "l":
    print ("Size: %s" % size)
    matplotlib.rcParams['font.size'] = 34
  
  #compute_rv(p)
  stats = load_log_files(e.get_log_root(), p['BallCatching']['strategy'], trial)
  print get_pretty_strategy_name(e)
  
  plot_position(e,stats,force_dim,title, filename_append)
#   plot_agent_velocity(e,stats,force_dim,title, filename_append)
#   plot_agent_acceleration(e,stats,force_dim,title, filename_append)
#   plot_viewing_angle(e,stats,title, filename_append)
#   plot_velocity_agent_and_vva(e,stats,force_dim,title, filename_append)
  
# --------------------------------------------------------------


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("experiment_name", type=str, )
  parser.add_argument("--trial", type=int, default=1,  help="number of trials per run")
  parser.add_argument("--force_dim", type=int, choices=[2,3], default=None, help="force 2D or 3D plots")
  parser.add_argument("--title", type=str, default=None)
  parser.add_argument("--size", type=str, choices=["m", "l"], default="l")
  parser.add_argument("--filename_append", type=str,default="")
  parser.add_argument("--noplot", action="store_true", default=False,  help="set to true if no plots wanted")
  args = parser.parse_args()

  if args.experiment_name[0] != ".":
    p = os.path.join(data_root, args.experiment_name)
  else:
    p = args.experiment_name
  
  e = Experiment(p)
  
  print "Plotting trial %d" % args.trial
  if args.force_dim is not None:
    print "Enforcing %d-D plot" % args.force_dim

  plot(e, args.trial, args.force_dim, args.title, args.size, args.filename_append)

  if not args.noplot:
    plt.show()
