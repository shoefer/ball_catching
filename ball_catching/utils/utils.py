#!/usr/bin/python

import os
import os.path
import datetime
import shutil
import yaml
import numpy as np
import sys

import shelve
import pickle

from ball_catching.config import data_root, params_yml
from ball_catching import dictionaries

import itertools


###############################################
def generate_timestamp():
  return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
  
def create_dictionary(name, content):
  dictionaries[name] = content

def update_dictionary(new_dictionary, old_dictionary=dictionaries, verbose=False, create_or_update=True):
  for name, content in new_dictionary.iteritems():
    update_dictionary_entry(name, content, old_dictionary, verbose)

def update_dictionary_entry(name, content, old_dictionary=dictionaries, verbose=False, create_or_update=True):
  if not (name in old_dictionary) and not create_or_update:
    print ("WARN not in dictionary:" + str(name))
    return
  
  if not (name in old_dictionary):
    old_dictionary[name] = {}
  
  for key, value in content.iteritems():
    old_dictionary[name][key] = value
    if verbose:
      print ("%s:%s -> %s" % (name, key, str(value)))
  
def merge_dictionaries(dicts=dictionaries):
  full = {}
  for name,d in dicts.iteritems():
    for k,v in d.iteritems():
      full["%s/%s" % (name, k)] = v
  return full

def make_flat(hier):
  return merge_dictionaries(hier)

def make_hierarchical(raw):
  dictionary = {}
  
  for k,v in raw.iteritems():
    items = k.split("/")
    assert len(items) == 2
    if items[0] not in dictionary:
      dictionary[items[0]] = {}
    dictionary[items[0]][items[1]] = v

  return dictionary

def load_yaml(filename):
  stream = open(filename, "r")
  raw = yaml.load(stream)
  
  try:
    return make_hierarchical(raw)
  except:
    # might already be hierarchical
    return raw


def str2paramList(s):
  return s.split("/")

def paramList2str(lst):
  return "/".join(map(str, lst))

def write_parameter_file(log_folder, dicts, yml_fn=params_yml):
   p_fn = os.path.join(log_folder, yml_fn)
   with open(p_fn, 'w') as outfile:
     outfile.write( yaml.dump(dicts, default_flow_style=True) )



#-----------------------------------------------------------------------

def param_range_dict_iterator(parameter_ranges, hierarchical=True):
  """ input: dictionary <parameter name> -> <list of params values>
      output: {
          (<parameter name1> -> value, <parameter name2> -> value), ...
        {"""
  if parameter_ranges is None or len(parameter_ranges) == 0:
    yield {}
  else:
    for p in itertools.product(* parameter_ranges.values()):
        res = dict(zip(parameter_ranges.keys(), p))
        if hierarchical:
          yield make_hierarchical(res)
        else:
          yield res

###############################################
class Experiment:
  def __init__(self, log_root, experiments_root=None, experiments_instance_name=None):
    self.log_root = log_root
    self.experiments_root = experiments_root
    self.experiments_instance_name = experiments_instance_name
    self.dictionary = None
    self.varying_parameters = None
    self.non_avg_varying_parameters = None
  
  def get_log_root(self):
    return self.log_root
  
  def _load_params(self):
    self.dictionary = load_yaml(os.path.join(self.log_root,params_yml))
    
    
  def get_params(self):
    if self.dictionary == None:
      try:
        self._load_params()
      except IOError, e:
        print (e)
        print ("  ---- skipping, probably additional folder or self-reference link")
        return None
        
    return self.dictionary

  def load_log(self, log_file):
    return np.loadtxt(os.path.join(self.log_root, log_file))

  def set_varying_parameter_assignment(self, p):
    self.varying_parameters = p

  def get_varying_parameter_assignment(self):
    return self.varying_parameters 

  def get_varying_parameters(self):
    """ p['BallCatching']['varying_parameters'] 
      should be a list of 2-tuples, containing 
      ('category/parameter/field', label')
      where 'field' is an optional entry number of a tuple type
      
      Returns a list which has the first element split up, e.g.
      ( [category, parameter, int(field)], label')
      """
    
    p = self.get_params()
    assert 'BallCatching' in p
    assert 'varying_parameters' in p['BallCatching']
    vp = p['BallCatching']['varying_parameters']
    
    params = []
    for v, label in vp:
      tup = v.split("/")
      if len(tup) == 3:
        tup[2] = int(tup[2])
      params.append ( (tup, label) )

    return params


###############################################

def load_experiments(experiment_name):
  es = []
  for d in os.listdir(os.path.join(data_root,experiment_name)):
    f = os.path.join(data_root,experiment_name,d)
    if not os.path.isdir(f):
      continue
    es.append(Experiment(f, experiment_name, d))

  return es


def get_stats_for_experiment(e):
  trials = e.get_params()["BallCatching"]["trials"]

  try:
    d = []
    v = []
    ce = []
    duration = []
    
    for trial in range(1,trials+1):
        try:
          a = e.load_log("agent_%d.log" % trial)
          #b = e.load_log("ball_%d.log" % trial)
          bc = e.load_log("ball_catching_%d.log" % trial)
        except Exception, e: 
          print "warn: could not open trial %d" % trial
          print e
          continue
        
        #tf = np.argmax(bc[:,1]==1)
        #dist = np.sqrt ((a[tf,1]-b[tf,1])**2 + (a[tf,3]-b[tf,3])**2)

        T = bc.shape[0]-3 + np.argmin(bc[-3:,2]) #bc[-1,0]
        
        # TERMINAL DISTANCE
        # calculate the minimum distance in step before ground contact, step 
        # after ground contact and interpolated distance between the steps
        #tf = bc[-1,0]
        dist =  bc[T,2] #bc[-1,2]

        if np.isnan(dist):
          print "WARN: distance is NaN"
        else:
          d.append(dist)

        # TERMINAL VELOCITY
        #print e.get_log_root()
        #print a[T,4]
        #print a[T,6]
        
        vel =  np.sqrt(a[T,4]**2 + a[T,6]**2) #bc[-1,2]

        if np.isnan(vel):
          print "WARN: velocity is NaN"
        else:
          v.append(vel)

        # CONTROL EFFORT
        #print a[:T,7:10]
        ceff = np.sqrt(np.sum(a[:T,7:10]**2))
        #print ceff

        if np.isnan(ceff):
          print "WARN: control effort is NaN"
        else:
          ce.append(ceff)

        # TRAJECTORY DURATIOn
        #print a[:T,7:10]
        t = bc[T,0]

        if np.isnan(t):
          print "WARN: duration is NaN"
        elif t == 0:
          raise Exception("duration is 0")
        else:
          duration.append(t)

    d_mean = np.mean(d)
    d_std = np.std(d)
    v_mean = np.mean(v)
    v_std = np.std(v)
    ce_mean = np.mean(ce)
    ce_std = np.std(ce)
    time_mean = np.mean(duration)
    time_std = np.std(duration)

  except Exception, e:
    print e
    return np.nan

  return d_mean, d_std, v_mean, v_std, ce_mean, ce_std, time_mean, time_std

###############################################
## -------------
# shelve/pickle stats saving and loading


def shelve_to_pickle(infile, outfile=None):
  db = shelve.open(infile)
  
  if outfile is None:
    outfile = infile+".pkl"

  # we store line by line in order not to have to load
  # the entire shelve into memory
  with open(outfile, 'wb') as outfile:
      print "shelve file has len: %d " % len(db)
      pickle.dump(len(db), outfile)
      for i,p in enumerate(db.iteritems()):
          pickle.dump(p, outfile)
          if i % 1000 == 0:
              sys.stdout.write('.')
      print "done"

def load_pickled_shelve(infile):
  db = {}
  with open(infile, 'rb') as infile:
    for i, _ in enumerate(xrange(pickle.load(infile))):
        k, v = pickle.load(infile)
        db[k] = v
        if i % 1000 == 0:
            sys.stdout.write('.')
    print "done"
        
  return db


def _load_db_from_shelve(experiment_root, db_name="compressed_log"):
    try:
        db = shelve.open(os.path.join(experiment_root, db_name))
    except:
        db = load_pickled_shelve(os.path.join(experiment_root, db_name+".pkl"))
    return db

def load_from_shelve(varying_params, experiment_root, db = None, db_name="compressed_log"):
    import pandas as pd

    if db is None:
        db = _load_db_from_shelve(experiment_root, db_name)
        
    # get varying params
    stats_len = len(STATS)
    df_raw = np.zeros( (len(db), len(varying_params) + stats_len))
    df_raw = np.cast['object'](df_raw)
    label_tuples = [ p.split('/') for p in varying_params ]

    param_list = []

    idx = 0
    for k, v in db.items():
        param_list.append (v['params'])
        
        # param settings
        for i,ltup in enumerate(label_tuples):
            assert (len(ltup) == 2)
            df_raw[idx, i] = v['params'][ltup[0]][ltup[1]]
             

        # stats
        for j,st in enumerate(STATS):
            df_raw[idx, j+i+1] = v['stats'][j]
            #print st + (" -> %d,%d -> " % (idx, j+i+1)) + str(v['stats'][j])
            #print df_raw[idx, j+i+1]

        idx += 1


    df = pd.DataFrame(df_raw, columns=varying_params + STATS)
        
    return df, param_list


def load_from_argmin_pickles(experiment_root):
    """ 
      Deprecated: load what argmin_parameters saves.
      
      Rather use what ball_catching_old.py saves if you store the compressed log
    """
    
    import pandas as pd
  
    with open(os.path.join(experiment_root, "varparam2exp.pkl"), "r") as f:
        E = pickle.load(f)
    with open(os.path.join(experiment_root, "varparam2stats.pkl"), "r") as f:
        D = pickle.load(f)
    with open(os.path.join(experiment_root, "labels.pkl"), "r") as f:
        l = pickle.load(f)
    #return E, D, l
    
    # distance data frame
    df = pd.DataFrame([ list(k) + d.tolist() for k,d in D.items()], columns=l + STATS)

    return df


#--------------------------------------------
def mpl_escape_latex(x):
    s = ""
    math_open = False
    for i, c in enumerate(x):
      if c == "$":
        math_open = not math_open
        
      if not math_open and c == "_":
        s += "\\"
      
      s += c
      
    return s
    
#--------------------------------------------

import subprocess

def pdfcrop(path):
    import distutils.spawn
    if not distutils.spawn.find_executable("pdfcrop"):
        import warnings
        warnings.warn("pdfcrop not installed; pdf outputs will have white borders")
        return False

    call = ["pdfcrop", path, path  ]
    print " ".join(call)
    pipe = subprocess.Popen(call)
    return pipe



