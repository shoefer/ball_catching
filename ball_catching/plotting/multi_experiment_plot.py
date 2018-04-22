#!/usr/bin/python

###############################################
# Plotting: Multiple experiments

import os
import os.path
import datetime
import shutil
import yaml
import numpy as np
import sys
from ball_catching.utils.slugify import slugify

import matplotlib
#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = [\
    #r"\usepackage{accents}",]
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rcParams['font.size'] = 24

#multi_figsize = (10,10)
multi_figsize = (10,7)


#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman'], 'size': 30})

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pylab as plt

from ball_catching.config import data_root
from ball_catching.utils.utils import pdfcrop, load_experiments, paramList2str

import argparse


###############################################
def collect_varying_parameters(experiments, constant_varying_params=[]):
  X = set()
  Z = set()
  AVG = set()
  
  varparam2exp = {}
  
  printed_warning=False

  print "Collecting parameters"
  for i, e in enumerate(experiments):
    if i>0 and i % 500 == 0:
      print "experiment %d" % i
    
    p = e.get_params()
    if p is None:
      continue

    all_varying_params = e.get_varying_parameters()
    framerate = p['BallCatching']['framerate']
    
    assert len(all_varying_params) >= 2

    # default: take first two params to be varying, average over rest
    varying_params = all_varying_params[:2]
    avg_params = all_varying_params[2:]
    
    if len(constant_varying_params) > 0:
      assert len(constant_varying_params) == 2
      varying_params = []
      avg_params = all_varying_params[:] # copy
      
      param_strs = dict( [ (paramList2str(x[0]), x) for x in all_varying_params] )
      for k in constant_varying_params:
        assert k in param_strs.keys()
        varying_params.append(param_strs[k])
        avg_params.remove(param_strs[k])

    if len(all_varying_params) > 2 and not printed_warning:
      print "There are %d varying params. " \
             "Vary %s and %s and compute average over %s" % (
             len(all_varying_params), 
             paramList2str(varying_params[0][0]),
             paramList2str(varying_params[1][0]),
             ", ".join( [paramList2str(x[0]) for x in avg_params ] ),
             )
      printed_warning = True


    e.non_avg_varying_parameters = varying_params

    labels = []
    
    tup, label = varying_params[0]
    x = p[tup[0]][tup[1]]
    if len(tup) == 3:
      x = x[tup[2]]
    X.add(x)
    labels.append(label)

    tup, label = varying_params[1]
    z = p[tup[0]][tup[1]]
    if len(tup) == 3:
      z = z[tup[2]]
    Z.add(z)
    labels.append(label)

    #print labels[0] + " " + str(x)
    #print labels[1] + " " + str(z)
    
    for tup, label in avg_params:
      avg = p[tup[0]][tup[1]]
      if len(tup) == 3:
        avg = avg[tup[2]]
      AVG.add(avg)
      #print label + " " + str(avg)
    
    #print "---"
    e.set_varying_parameter_assignment( (x,z) )
    if (x,z) not in varparam2exp:
      varparam2exp[(x,z)] = []
    varparam2exp[(x,z)].append(os.path.join(e.experiments_root, e.experiments_instance_name))
    
  X = sorted(list(X))
  Z = sorted(list(Z))
  AVG = sorted(list(AVG))

  for x in X:
    for z in Z:
      print " (%.2f,%.2f) --> " % (x, z)
      for s in varparam2exp[x,z]:
        print "     %s" % s

  
  return X,Z,labels,AVG


###############################################
def plot_surface(fig, ax, title, X, Y, Z, labels, linewidth=0.2, vlim=None, zlim=[None,None], contour=False, elev=20, azi=45):
  if vlim is None:
    vlim = zlim
  ax.set_title(title)
  #ax = fig.gca(projection='3d')

  xlabelpad, ylabelpad, zlabelpad = [35,35,10]

  if zlim[1] is not None:
    Z[np.where(Z > zlim[1])] = zlim[1]
  
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, vmin=vlim[0], vmax=vlim[1], 
        linewidth=linewidth, antialiased=False, )  
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
  ax.set_xlabel(labels[0], labelpad=xlabelpad)
  ax.set_ylabel(labels[1], labelpad=ylabelpad)
  ax.set_zlabel(labels[2], labelpad=zlabelpad)


  if zlim[0] == 0 and zlim[1] == 5:
    # hacky: for agent_ball_distance
    ax.set_zticks([0,1,2,3,4,5])
    ax.set_zticklabels(map (lambda x: "$%d$" % x, [0,1,2,3,4,5]))

  # reformat ticklabels
  for ticks_and_labels, setter, in zip([ zip(ax.get_xticks(), ax.get_xticklabels()),
                                        zip(ax.get_yticks(), ax.get_yticklabels()),
                                        zip(ax.get_zticks(), ax.get_zticklabels())],
                                        (ax.set_xticklabels, ax.set_yticklabels, ax.set_zticklabels)):
    ticklbls = []
    fnt_size = "huge"

    tkz = zip(*ticks_and_labels)[0]
    tick_range =  np.max(tkz) - np.min(tkz)

    for tick, lbl in ticks_and_labels:
      #txt = lbl.get_text()
      #if txt == "":
      if tick_range <= 1.5:
        tl = "%.1f" % tick
      else:
        tl = "%d" % tick
      txt = r"\%s{$%s$}" % (fnt_size, tl)
      #else:
      #  txt = r"\%s{%s}" % (fnt_size, tick)
      lbl.set_text(txt)
      ticklbls.append(lbl)
    setter(ticklbls)

  # move y ticks a bit to the left, so the -15 does not collide
  [t.set_ha('right') for t in ax.get_yticklabels()]

  vmin = 0.
  vmax = 1.
  
  ax.view_init(elev, azi)
  
  if contour:
    if np.min(Z) != np.max(Z):
      cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)
    if np.min(X) != np.max(X):
      #cset = ax.contour(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.coolwarm)
      cset = ax.contour(X, Y, Z, zdir='x', offset=np.max(X), cmap=cm.coolwarm) # project to opposite side
    if np.min(Y) != np.max(Y):
      #cset = ax.contour(X, Y, Z, zdir='y', offset=np.min(Y), cmap=cm.coolwarm)
      cset = ax.contour(X, Y, Z, zdir='y', offset=np.max(Y), cmap=cm.coolwarm) # project to opposite side
  
  if zlim != [None,None]:
    ax.set_zlim3d(zlim[0], zlim[1])
  fig.colorbar(surf, shrink=0.75, aspect=20)

  return ax

###############################################
def plot_agent_ball_distance(experiment_name, experiments, X, Z, labels, AVG, title=None, plot=True, copy_pdf_to=None):
  distances = {}
  flight_times = {}

  # for calculating variance if 
  #distances_sq = {}
  #flight_times_sq = {}

  #avg_count = {}

  actual_trials_global = None

  print "***Plotting agent ball distance***"
    
  for i,e in enumerate(experiments):
    if i>0 and i % 500 == 0:
      print "experiment %d" % i
    
    p = e.get_params()
    if p is None:
      continue
    
    x,z = e.get_varying_parameter_assignment()
    
    #x_init = p['Agent']['x_init']
    framerate = p['BallCatching']['framerate']
    trials = p['BallCatching']['trials']
    
    actual_trials = trials
    
    for trial in range(1,trials+1):

      try:
        a = e.load_log("agent_%d.log" % trial)
        b = e.load_log("ball_%d.log" % trial)
        bc = e.load_log("ball_catching_%d.log" % trial)
      except Exception,excep:
        print excep
        continue
      
      #tf = np.argmax(bc[:,1]==1)
      #dist = np.sqrt ((a[tf,1]-b[tf,1])**2 + (a[tf,3]-b[tf,3])**2)
      
      #bc[-2:,2] = 1000
      # calculate the minimum distance in step before ground contact, step 
      # after ground contact and interpolated distance between the steps
      T = bc.shape[0]-3 + np.argmin(bc[-3:,2]) #bc[-1,0]
      tf = bc[-1,0]
      dist =  bc[T,2] #bc[-1,2]

      if np.isnan(dist):
        print "WARN: distance is NaN"

      if (x, z) in distances:
        #assert (x,z) in avg_count
        assert (x,z) in flight_times
        #assert (x,z) in distances_sq
        #assert (x,z) in flight_times_sq
      else:
        distances [(x, z)] = [] #0
        flight_times [(x, z)] = [] #0
        #distances_sq [(x, z)] = 0
        #flight_times_sq [(x, z)] = 0
        #avg_count[(x,z)] = 0
      
      distances [(x, z)].append(dist)
      flight_times [(x, z)].append(tf)
      #distances [(x, z)] += dist
      #flight_times [(x, z)] += tf
      #distances_sq [(x, z)] += dist * dist
      #flight_times_sq [(x, z)] += tf * tf
      #avg_count[(x,z)] += 1

    if actual_trials_global is not None and actual_trials_global != actual_trials:
      raise ("ERROR: number of trials is not consistent!")
    actual_trials_global = actual_trials

  if actual_trials != trials:
    print "WARN: less trials than expected. Expected %d got %d" % (trials, actual_trials)

  if len(distances.values()[0]) != actual_trials:
    print "WARN: more values per iteration than expected; assuming additional varying parameters"
    actual_trials = len(distances.values()[0])
    

  #print "avg_count %s" % str(avg_count)

  # create numpy arrays
  Y = np.zeros( (len(X), len(Z), actual_trials) ) 
  #Ystd = np.zeros( (len(X), len(Z)) ) 
  W = np.zeros( (len(X), len(Z), actual_trials) ) 
  #Wstd = np.zeros( (len(X), len(Z)) ) 
  for i,x in enumerate(X):
    for j,z in enumerate(Z):
      try:
        Y[i,j] = distances[(x, z)] #/ avg_count[(x,z)]
        W[i,j] = flight_times[(x, z)] #/ avg_count[(x,z)]
        
        #if avg_count[(x,z)] > 1:
          #Ystd[i,j] = np.sqrt((distances_sq[(x, z)]/ avg_count[(x,z)] - (distances[(x, z)]/ avg_count[(x,z)])**2))
          #Wstd[i,j] = np.sqrt((flight_times_sq[(x, z)]/ avg_count[(x,z)] - (flight_times[(x, z)]/ avg_count[(x,z)])**2))
            
      except KeyError as e:
        print "[WARN] %s" % e
        Y[i,j] = 0.0
        W[i,j] = 0.0
        Ystd[i,j] = 0.0
        Wstd[i,j] = 0.0
        
  Ymean=Y.mean(axis=2).T
  Ystd=Y.std(axis=2).T
  Wmean=W.mean(axis=2).T
  Wstd=W.std(axis=2).T

  print ("Mean terminal distance: ")
  print Ymean.mean()
  
  X, Z = np.meshgrid(X, Z)

  #####
  # write matrices to files
  print "Writing matrices to files"
  nvp_names = [ paramList2str(x[0]) for x in e.non_avg_varying_parameters ]
  xpath = os.path.join(data_root, e.experiments_root, slugify("agent_ball_distance_%s-%s_X" % tuple(nvp_names) ) + ".txt")
  zpath = os.path.join(data_root, e.experiments_root, slugify("agent_ball_distance_%s-%s_Z" % tuple(nvp_names) ) + ".txt")

  ypath = os.path.join(data_root, e.experiments_root, slugify("agent_ball_distance_%s-%s_Y" % tuple(nvp_names) ) + ".npy")
  ymeanpath = os.path.join(data_root, e.experiments_root, slugify("agent_ball_distance_%s-%s_Ymean" % tuple(nvp_names) ) + ".txt")
  ystdpath = os.path.join(data_root, e.experiments_root, slugify("agent_ball_distance_%s-%s_Ystd" % tuple(nvp_names) ) + ".txt")
  
  wpath = os.path.join(data_root, e.experiments_root, slugify("agent_ball_distance_%s-%s_W" % tuple(nvp_names) ) + ".npy")
  wmeanpath = os.path.join(data_root, e.experiments_root, slugify("agent_ball_distance_%s-%s_W" % tuple(nvp_names) ) + ".txt")
  wstdpath = os.path.join(data_root, e.experiments_root, slugify("agent_ball_distance_%s-%s_Wstd" % tuple(nvp_names) ) + ".txt")
  
  np.savetxt(xpath, X)
  np.savetxt(zpath, Z)
  
  np.save(ypath, Y)
  np.savetxt(ymeanpath, Ymean)
  np.savetxt(ystdpath, Ystd)

  np.save(wpath, W)
  np.savetxt(wmeanpath, Wmean)
  np.savetxt(wstdpath, Wstd)

  #####
  if plot:
    print "Plotting"
    
    fig = plt.figure(figsize=multi_figsize)

    #### distance
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    #zlim =(0, np.max([5,np.max(Y)]))
    #zlim =(0, np.max([5,np.max(Y)]))
    zlim =(0., 5.)
    vlim =(0., 1.)
   
    # add standard deviation (will be plotted as transparent surface)
    #if np.max(avg_count) > 1:
    if actual_trials > 1:
      Y_p_std = Ymean+Ystd
      Y_m_std = Ymean-Ystd
      if zlim[1] is not None:
        Y_p_std[np.where(Y_p_std > zlim[1])] = zlim[1]
      if zlim[1] is not None:
        Y_m_std[np.where(Y_m_std > zlim[1])] = zlim[1]
        Y_m_std[np.where(Y_m_std < zlim[0])] = zlim[0]

      ax.plot_surface(X, Z, Y_p_std, alpha=0.1, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0.0, antialiased=True)  
      ax.plot_surface(X, Z, Y_m_std, alpha=0.1, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0.0, antialiased=True)  
    else:
      # add transparent -1 / 1 error planes
      err1 = np.ones( Ymean.shape ) 
    
    filename = "agent_ball_distance.pdf"
    s_title = "%s"
    #s_title = "%s - terminal distance"
    if title is not None:
      title = s_title % title
      filename = slugify(title) + "_" + filename
      # latex stuff
      title = r"\begin{center}" + title + r"\end{center}"
    else:
      title = s_title % "/".join(experiment_name.split("/")[-1:])
      title = title.replace("_", "-")[:30]
      
    print title
    
    new_label_dict = {"$a_x$": r"\begin{center}initial distance to \\impact point $D$ \end{center}",
                      "$V$": r"\begin{center}initial ball \\ velocity $\nu$ \end{center}",
                      "$D$":  r"\begin{center}initial distance to \\ impact point $D$ \end{center}" ,
#                      "$\phi$":  r"\phi" ,
                      }
    new_labels = [ new_label_dict[l]  if l in new_label_dict else l for l in labels ]
    
    plot_surface(fig, ax, title, X, Z, Ymean, new_labels + [r"\textbf{terminal distance}"], zlim=zlim, vlim=vlim, contour=True, elev=15, azi=-135)
    #ax.xaxis._axinfo['label']['space_factor'] = 2.5
    #ax.yaxis._axinfo['label']['space_factor'] = 2.5
    #ax.zaxis._axinfo['label']['space_factor'] = 2.0

    plt.tight_layout()
    
    distfigpath = os.path.join(data_root, e.experiments_root, filename)
    print " Saving %s" % distfigpath
    plt.savefig(distfigpath, format='pdf')
    pdfcrop(distfigpath)

    if copy_pdf_to is not None:
      p = os.path.join(copy_pdf_to, filename)
      print " Copying to %s" % p
      try:
        plt.savefig(p, format='pdf')
      except IOError, e:
        print "FAILED!"
        print e


    # FIXME
    return

    #### flight time

    #ax = fig.add_subplot(1, 2, 2, projection='3d')
    fig = plt.figure(figsize=multi_figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    s_title = "%s - flight time"
    if title is not None:
      title = s_title % title
    else:
      title = s_title % experiment_name
    
    plot_surface(fig, ax, title, X, Z, Wmean, labels + ["flight time"], contour=True)

    distfigpath = os.path.join(data_root, e.experiments_root, "flight_time.pdf")
    plt.savefig(distfigpath, format='pdf')
    pdfcrop(distfigpath)

  ###
  print "--------------------------"
  print "Average catching DISTANCE: %f" % np.mean(Y)


def plot_catch_velocity(experiment_name, experiments, X, Z, labels, AVG, plot=True, title=None):
  return plot_catch_velocity_and_control_effort(experiment_name, experiments, X, Z, labels, AVG, control_effort=False, plot=plot, title=title)


def plot_catch_velocity_and_control_effort(experiment_name, experiments, X, Z, labels, AVG, control_effort=True, plot=True, title=None, copy_pdf_to=None):
  velocities = {}
 
  print "***Plotting velocity at interception***"

  if control_effort:
    controleffs = {}
    print "***Plotting control effort***"
    
    
  actual_trials_global = None
    
  for i,e in enumerate(experiments):
    if i>0 and i % 500 == 0:
      print "experiment %d" % i

    p = e.get_params()
    if p is None:
      continue

    x,z = e.get_varying_parameter_assignment()
    
    #x_init = p['Agent']['x_init']
    framerate = p['BallCatching']['framerate']
    trials = p['BallCatching']['trials']
    
    actual_trials = trials
    
    for trial in range(1,trials+1):
      try:
        a = e.load_log("agent_%d.log" % trial)
        b = e.load_log("ball_%d.log" % trial)
        bc = e.load_log("ball_catching_%d.log" % trial)
      except Exception,excep:
        print excep
        actual_trials = trial
        break
      
      #tf = np.argmax(bc[:,1]==1)
      #dist = np.sqrt ((a[tf,1]-b[tf,1])**2 + (a[tf,3]-b[tf,3])**2)
      
      # calculate the minimum distance in step before ground contact, step 
      # after ground contact and interpolated distance between the steps
      T = bc.shape[0]-3 + np.argmin(bc[-3:,2]) #bc[-1,0]
      tf = bc[T,0]
      #print e.get_log_root()
      #print a[T,4]
      #print a[T,6]
      
      v =  np.sqrt(a[T,4]**2 + a[T,6]**2) #bc[-1,2]

      if (x, z) not in velocities:
        velocities [(x, z)] = []
        #avg_count[(x,z)] = 0
      velocities [(x, z)].append (v)
      #avg_count[(x,z)] += 1

      if control_effort:
        if (x, z) not in controleffs:
          controleffs[(x, z)] = []
        ce = np.sqrt(np.sum(a[:T,7:10]**2))
        controleffs[(x, z)].append (ce)

    if actual_trials_global is not None and actual_trials_global != actual_trials:
      raise ("ERROR: number of trials is not consistent!")
    actual_trials_global = actual_trials

  if actual_trials != trials:
    print "WARN: less trials than expected. Expected %d got %d" % (trials, actual_trials)

  if len(velocities.values()[0]) != actual_trials:
    print "WARN: more values per iteration than expected; assuming additional varying parameters"
    actual_trials = len(velocities.values()[0])

  Y = np.zeros( (len(X), len(Z), actual_trials) ) 
  for i,x in enumerate(X):
    for j,z in enumerate(Z):
      try:
        #Y[i,j] = velocities[(x, z)] / avg_count[(x,z)]
        Y[i,j] = velocities[(x, z)] #/ avg_count[(x,z)]
      except KeyError as e:
        print "[WARN] %s" % e
        Y[i,j] = np.nan #0.0
  Ymean=Y.mean(axis=2).T
  #print Y

  if control_effort:
    C = np.zeros( (len(X), len(Z), actual_trials) ) 
    for i,x in enumerate(X):
      for j,z in enumerate(Z):
        try:
          C[i,j] = controleffs[(x, z)] #/ avg_count[(x,z)]
        except KeyError as e:
          print "[WARN] %s" % e
          C[i,j] = np.nan #0.0
    Cmean=C.mean(axis=2).T
    #print Y

  #####
  X, Z = np.meshgrid(X, Z)
  
  #####
  # write matrices to files
  print "Writing matrices to files"
  nvp_names = [ paramList2str(x[0]) for x in e.non_avg_varying_parameters ]
  xpath = os.path.join(data_root, e.experiments_root, slugify("agent_velocity_%s-%s_X" % tuple(nvp_names) ) + ".txt")
  zpath = os.path.join(data_root, e.experiments_root, slugify("agent_velocity_%s-%s_Z" % tuple(nvp_names) ) + ".txt")
  
  ypath = os.path.join(data_root, e.experiments_root, slugify("agent_velocity_%s-%s_Y" % tuple(nvp_names) ) + ".npy")
  ymeanpath = os.path.join(data_root, e.experiments_root, slugify("agent_velocity_%s-%s_Ymean" % tuple(nvp_names) ) + ".txt")
  #ystdpath = os.path.join(data_root, e.experiments_root, slugify("agent_velocity_%s-%s_Ystd" % tuple(nvp_names) ) + ".txt")
  np.savetxt(xpath, X)
  np.savetxt(zpath, Z)

  np.save(ypath, Y)
  np.savetxt(ymeanpath, Ymean)
  #np.savetxt(ystdpath, Ystd)
  
  if plot:
    fig = plt.figure(figsize=(multi_figsize[0]*2, multi_figsize[1]))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    s_title = "%s - terminal velocity"
    if title is not None:
      title = s_title % title
    else:
      title = s_title % "/".join(experiment_name.split("/")[-1:])
      title = title.replace("_", "-")[:30]
      
    plot_surface(fig, ax, title, X, Z, Ymean, labels + ["$m/s$"])

    filename = "catch_velocity.pdf"
    cvfigpath = os.path.join(data_root, e.experiments_root, filename)
    plt.savefig(cvfigpath, format='pdf')
    pdfcrop(cvfigpath)

    if copy_pdf_to is not None:
      p = os.path.join(copy_pdf_to, filename)
      print " Copying to %s" % p
      try:
        plt.savefig(p, format='pdf')
      except IOError, e:
        print "FAILED!"
        print e

  #################
  if control_effort:
    print "Writing control effort matrices to files"
    xpath = os.path.join(data_root, e.experiments_root, slugify("agent_controleffort_%s-%s_X" % tuple(nvp_names) ) + ".txt")
    zpath = os.path.join(data_root, e.experiments_root, slugify("agent_controleffort_%s-%s_Z" % tuple(nvp_names) ) + ".txt")
    
    cpath = os.path.join(data_root, e.experiments_root, slugify("agent_controleffort_%s-%s_Y" % tuple(nvp_names) ) + ".npy")
    cmeanpath = os.path.join(data_root, e.experiments_root, slugify("agent_controleffort_%s-%s_Ymean" % tuple(nvp_names) ) + ".txt")
    #ystdpath = os.path.join(data_root, e.experiments_root, slugify("agent_velocity_%s-%s_Ystd" % tuple(nvp_names) ) + ".txt")
    np.savetxt(xpath, X)
    np.savetxt(zpath, Z)

    np.save(cpath, C)
    np.savetxt(cmeanpath, Cmean)

    if plot:
      fig = plt.figure(figsize=(multi_figsize[0]*2, multi_figsize[1]))

      ax = fig.add_subplot(1, 1, 1, projection='3d')
      s_title = "%s - control effort"
      
      title = s_title % "/".join(experiment_name.split("/")[-1:])
      title = title.replace("_", "-")[:30]
      
      plot_surface(fig, ax, title, X, Z, Cmean, labels + ["$m/s$"])

      filename = "control_effort.pdf"
      cvfigpath = os.path.join(data_root, e.experiments_root, filename)
      plt.savefig(cvfigpath, format='pdf')
      pdfcrop(cvfigpath)

      if copy_pdf_to is not None:
        p = os.path.join(copy_pdf_to, filename)
        print " Copying to %s" % p
        try:
          plt.savefig(p, format='pdf')
        except IOError, e:
          print "FAILED!"
          print e
 
  ###
  print "--------------------------"
  print "Average catching VELOCITY: %f" % np.mean(Y)

  if control_effort:
    print "--------------------------"
    print "Average CONTROL EFFORT: %f" % np.mean(C)


def plot_catch_interpolation_error(experiment_name, experiments, X, Z, labels, AVG):
  ip_error = {}
  avg_count = {}

  print "***Plotting interpolation error at interception***"
    
  for i,e in enumerate(experiments):
    if i>0 and i % 500 == 0:
      print "experiment %d" % i

    p = e.get_params()
    if p is None:
      continue

    x,z = e.get_varying_parameter_assignment()
    
    #x_init = p['Agent']['x_init']
    framerate = p['BallCatching']['framerate']
    trials = p['BallCatching']['trials']

    for trial in range(1, trials+1):
    
      a = e.load_log("agent_%d.log" % trial)
      b = e.load_log("ball_%d.log" % trial)
      bc = e.load_log("ball_catching_%d.log" % trial)
      
      if (x, z) not in ip_error:
        ip_error [(x, z)] = 0
        avg_count[(x,z)] = 0
      ip_error [(x, z)] += -bc[-1,0] + bc[-2,0]
      avg_count[(x,z)] += 1

  ip = np.zeros( (len(X), len(Z)) ) 
  for i,x in enumerate(X):
    for j,z in enumerate(Z):
      try:
        ip[i,j] = ip_error[(x, z)] / avg_count[(x, z)]
      except KeyError as e:
        print "[WARN] %s" % e
        ip[i,j] = 0.0
  ip=ip.T
  #print ip_error

  X, Z = np.meshgrid(X, Z)

  fig = plt.figure(figsize=(multi_figsize[0]*2, multi_figsize[1]))
  ax = fig.add_subplot(1, 1, 1, projection='3d')
  title = "%s - ip error"  % experiment_name
  plot_surface(fig, ax, title, X, Z, ip, labels + ["interpolation error"])
  
def plot_ball_impact_point(experiment_name, experiments, X, Z, labels, AVG):
  impact_points = {}

  # for calculating variance if 
  impact_points_sq = {}

  displacement = []

  avg_count = {}

  # assuming 1 trial
  trial = 1
    
  print "***Plotting agent ball distance***"
    
  for i,e in enumerate(experiments):
    if i>0 and i % 500 == 0:
      print "experiment %d" % i

    p = e.get_params()
    if p is None:
      continue
    
    x,z = e.get_varying_parameter_assignment()
    
    #x_init = p['Agent']['x_init']
    framerate = p['BallCatching']['framerate']
    
    b = e.load_log("ball_%d.log" % trial)

    x_disp = (b[-1,1]-b[0,1])
    z_disp = (b[-1,3]-b[0,1])
    dist =  np.sqrt( x_disp**2 + z_disp**2)

    displacement.append( (x_disp, z_disp) )

    if np.isnan(dist):
      print "WARN: distance is NaN"

    if (x, z) in impact_points:
      assert (x,z) in avg_count
      assert (x,z) in impact_points
      assert (x,z) in impact_points_sq
    else:
      impact_points [(x, z)] = 0
      impact_points_sq [(x, z)] = 0
      avg_count[(x,z)] = 0
      
    impact_points [(x, z)] += dist
    impact_points_sq [(x, z)] += dist * dist
    avg_count[(x,z)] += 1

  #print "avg_count %s" % str(avg_count)

  # create numpy arrays
  Y = np.zeros( (len(X), len(Z)) ) 
  Ystd = np.zeros( (len(X), len(Z)) ) 
  W = np.zeros( (len(X), len(Z)) ) 
  Wstd = np.zeros( (len(X), len(Z)) ) 
  for i,x in enumerate(X):
    for j,z in enumerate(Z):
      try:
        W[i,j] = impact_points[(x, z)] / avg_count[(x,z)]
        
        if avg_count[(x,z)] > 1:
          Wstd[i,j] = np.sqrt((impact_points_sq[(x, z)]/ avg_count[(x,z)] - (impact_points[(x, z)]/ avg_count[(x,z)])**2))
            
      except KeyError as e:
        print "[WARN] %s" % e
        Y[i,j] = 0.0
        W[i,j] = 0.0
        Ystd[i,j] = 0.0
        Wstd[i,j] = 0.0
        
  W=W.T
  Wstd=Wstd.T

  displacement = np.array(displacement)
  print displacement.shape

  #print Y

  X, Z = np.meshgrid(X, Z)

  #####
  print "Plotting"
  
  fig = plt.figure(figsize=(multi_figsize[0]*2, multi_figsize[1]))

  #### distance
  ax = fig.add_subplot(1, 2, 1, projection='3d')

  # add standard deviation (will be plotted as transparent surface)
  if np.max(avg_count) > 1:
    ax.plot_surface(X, Z, W+Wstd, alpha=0.1, rstride=1, cstride=1, cmap=cm.coolwarm,
          linewidth=0.0, antialiased=True)  
    ax.plot_surface(X, Z, W-Wstd, alpha=0.1, rstride=1, cstride=1, cmap=cm.coolwarm,
          linewidth=0.0, antialiased=True)  
  else:
    # add transparent -1 / 1 error planes
    err1 = np.ones( W.shape ) 
  
  title = "%s - ball impact distance"  % experiment_name
  plot_surface(fig, ax, title, X, Z, W, labels + ["ball impact"], contour=True)

  #######################################
  #fig = plt.figure(figsize=multi_figsize)
  ax = fig.add_subplot(1, 2, 2)
  ax.scatter(displacement[:,0], displacement[:,1])
  ax.set_title("Ball impact point")

  distfigpath = os.path.join(data_root, e.experiments_root, "ball_impact.pdf")
  plt.savefig(distfigpath, format='pdf')
  pdfcrop(distfigpath)

  
#def generate_drag_data_file(experiment_name, experiments, X, Z, labels, AVG):
  #ip_error = {}
  #avg_count = {}

  ## assuming 1 trial
  #trial = 1
  
  #filenameX = "/home/shoefer/.ros/data/drag_data_X.txt"
  #print "***Generate drag data file %s***" % filenameX

  #X = []
    
  #for i,e in enumerate(experiments):
    #if i>0 and i % 500 == 0:
      #print "experiment %d" % i

    #b = e.load_log("ball_%d.log" % trial)
    #for i in range(b.shape[0]):
      #X.append(b[i,[0,1,2,4,5]])
    
  #np.savetxt(filenameX, X)
    
def plot_cov_rv(experiment_name, experiments, X, Z, labels, AVG):  
  rvs = {}
  # for calculating variance if 
  rvs_sq = {}

  avg_count = {}

  # assuming 1 trial
  trial = 1
    
  print "***Plotting cov reference value***"
    
  for i,e in enumerate(experiments):
    if i>0 and i % 500 == 0:
      print "experiment %d" % i

    p = e.get_params()
    if p is None:
      continue

    x,z = e.get_varying_parameter_assignment()
    
    #x_init = p['Agent']['x_init']
    try:
      cov_log = e.load_log("COVStrategy_%d.log" % trial)
    except:
      print "Cannot load COVStrategy_%d.log - probably different strategy" % trial
      return

    cov = cov_log[-1,-1]

    if (x, z) in rvs:
      assert (x,z) in avg_count
      assert (x,z) in rvs
      assert (x,z) in rvs_sq
    else:
      rvs [(x, z)] = 0
      rvs_sq [(x, z)] = 0
      avg_count[(x,z)] = 0
      
    rvs [(x, z)] += cov
    rvs_sq [(x, z)] += cov * cov
    avg_count[(x,z)] += 1

  # create numpy arrays
  Y = np.zeros( (len(X), len(Z)) ) 
  Ystd = np.zeros( (len(X), len(Z)) ) 
  W = np.zeros( (len(X), len(Z)) ) 
  Wstd = np.zeros( (len(X), len(Z)) ) 
  for i,x in enumerate(X):
    for j,z in enumerate(Z):
      try:
        W[i,j] = rvs[(x, z)] / avg_count[(x,z)]
        
        if avg_count[(x,z)] > 1:
          Wstd[i,j] = np.sqrt((rvs_sq[(x, z)]/ avg_count[(x,z)] - (rvs[(x, z)]/ avg_count[(x,z)])**2))
            
      except KeyError as e:
        print "[WARN] %s" % e
        Y[i,j] = 0.0
        W[i,j] = 0.0
        Ystd[i,j] = 0.0
        Wstd[i,j] = 0.0
        
  W=W.T
  Wstd=Wstd.T

  X, Z = np.meshgrid(X, Z)

  #####
  print "Plotting"
  
  fig = plt.figure(figsize=(multi_figsize[0]*2, multi_figsize[1]))

  #### distance
  ax = fig.add_subplot(1, 1, 1, projection='3d')

  # add standard deviation (will be plotted as transparent surface)
  if np.max(avg_count) > 1:
    ax.plot_surface(X, Z, W+Wstd, alpha=0.1, rstride=1, cstride=1, cmap=cm.coolwarm,
          linewidth=0.0, antialiased=True)  
    ax.plot_surface(X, Z, W-Wstd, alpha=0.1, rstride=1, cstride=1, cmap=cm.coolwarm,
          linewidth=0.0, antialiased=True)  
  else:
    # add transparent -1 / 1 error planes
    err1 = np.ones( W.shape ) 
  
  title = "%s - reference value"  % experiment_name
  plot_surface(fig, ax, title, X, Z, W, labels + ["rv"], contour=True)

  distfigpath = os.path.join(data_root, e.experiments_root, "cov_rv.pdf")
  plt.savefig(distfigpath, format='pdf')
  pdfcrop(distfigpath)
    

def multi_experiment_plot(experiment_name, varying_non_averaged_params=[],
                          title=None, 
                          copy_pdf_to=None,
                          vel_and_ce=False):
  experiments = load_experiments(experiment_name)

  # assume varying params stays constant
  #X,Z,labels, AVG = collect_varying_parameters(experiments, sys.argv[2:])
  X,Z,labels, AVG = collect_varying_parameters(experiments, varying_non_averaged_params)

  print "Plotting"
  
  plot_agent_ball_distance(experiment_name, experiments, X, Z, labels, AVG, title=title, copy_pdf_to=copy_pdf_to)
  if vel_and_ce:
    plot_catch_velocity_and_control_effort(experiment_name, experiments, X, Z, labels, AVG, title=title, copy_pdf_to=copy_pdf_to)

  #plot_cov_rv(experiment_name, experiments, X, Z, labels, AVG, title=args.title)

  #plot_catch_interpolation_error(experiment_name, experiments, X, Z, labels, AVG)
  
  #plot_ball_impact_point(experiment_name, experiments, X, Z, labels, AVG)
  
  #generate_drag_data_file(experiment_name, experiments, X, Z, labels, AVG)

#######################################
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("experiment_name", type=str )
  parser.add_argument("varying_non_averaged_params", nargs='*', type=str, default=[])
  parser.add_argument("--title", type=str, help="title for plot", default=None, required=False)
  #parser.add_argument("--copy_pdf_to", type=str, help="optional ", default=None)
  parser.add_argument("--copy_pdf_to", type=str, help="optional ", required=False)
                      #default="/home/shoefer/Documents/pubs/papers/Hoefer-16-BC/figures/individual2D/", )
  parser.add_argument("--no_plot_show", action="store_true", default=False)
  parser.add_argument("--vel_and_ce", action="store_true", default=False, help="plot terminal velocity and control effort", required=False)
  args = parser.parse_args()
  
  if len(sys.argv) < 2:
    print "Usage: multi_experiment_plot.py <experiment_name> [<varying non-averaged params...>]"

  multi_experiment_plot(args.experiment_name, args.varying_non_averaged_params,
                        args.title, args.copy_pdf_to, args.vel_and_ce)

  if not args.no_plot_show:
    plt.show()
