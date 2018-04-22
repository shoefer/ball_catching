#!/usr/bin/python

###############################################
# Plotting: Compare multiple strategies 

import os
import yaml
import numpy as np
import sys
import argparse
import pickle
import pandas as pd
from collections import OrderedDict
import re

from warnings import warn

import matplotlib
from matplotlib import rc

#font_rc = {'size': 20}
font_rc = {'size': 30}
#font_rc['sans-serif'] = font_rc['serif']

rc_plus = {'axes.labelsize': font_rc['size'],
           'axes.titlesize':  font_rc['size'],
           'legend.fontsize':  font_rc['size'],
           'xtick.labelsize':  font_rc['size'],
           'ytick.labelsize':  font_rc['size'],
           'figure.autolayout': True,
}
rc_plus.update(dict([ ( 'font.'+k, v ) for k,v in font_rc.items() ]))
rc(rc_plus)
rc('font',**font_rc)

# use autolayout
matplotlib.rcParams['figure.autolayout'] = True

import matplotlib.pylab as plt
import seaborn.apionly as sns

from ball_catching.utils.slugify import slugify
from ball_catching.config import data_root



meta_ptn_str = "agent_%s_agentx_init[0-9]-ballv_init[0-9]_([a-zA-Z]+)\.([a-zA-Z]+)"

dist_ptn_str = meta_ptn_str % "ball_distance"
dist_ptn = re.compile(dist_ptn_str)
velocity_ptn_str = meta_ptn_str % "velocity"
velocity_ptn = re.compile(velocity_ptn_str)
controleffort_ptn_str = meta_ptn_str % "controleffort"
controleffort_ptn = re.compile(controleffort_ptn_str)

strategy_ptn_str = r"/([A-Za-z0-9 ]+)_[0-9]D.+"
strategy_ptn = re.compile(strategy_ptn_str)

res_types = ["distance", "velocity", "controleffort"]

image_format = "pdf"

OMIT_LQR = True
THREE_D = False

#=====================================================

pretty_strategy_names_cpp = OrderedDict( (
  # CPP
  ("KalmanLQRStrategy", "LQG"),
  ("LQRStrategy", "LQR only"),
  ("KalmanLQRV10Strategy", "LQG (vel=0.01)"),
  ("LQRV10Strategy", "LQR only (vel=0.01)"),
  ("KalmanLQRV1000Strategy", "LQG (vel=1)"),
  ("LQRV1000Strategy","LQR only (vel=1)"),

  ("ELQRStrategy", "iLQR"),
  ("ELQRSoccerStrategy", "iLQR (soccer)"),

  ("OACStrategy", "OAC"),
  ("JerkOACStrategy", "OAC-jerk"),
  ("ConstCOVStrategy", "COV"),
  ("ObsInitCOVStrategy", "COV-IO"),
  ("OACCOVStrategy", "COV-OAC"),

  ("OACCOVStrategy3D", "COV-OAC +CBA"),
  ("OAC3DStrategy", "OAC +CBA"),
))

pretty_strategy_names_py = OrderedDict( (
  # Python
  # I don't know why but this has to be different from CPP labels
  ("LQRStrategy", "LQR only"),
  ("LQGStrategy", "LQG"),
  ("LQGV10Strategy", r"LQG {\huge (v=0.01)}"),
  ("LQGV1000Strategy", r"LQG {\huge (v=1)}"),
  ("iLQRStrategy", "iLQR only"),
  ("iLQGStrategy", "iLQG"),
  ("iLQGSoccerStrategy", r"iLQG {\huge (soccer)}"),

  #("MPCStrategyDefault", "MPC"),
  ("MPCStrategy", "MPC"),
  ("MPCStrategyHighUncertainty", r"MPC {\huge (Unc.)}"),

  ("OAC2DStrategy", "OAC"),
  ("COVConst2DStrategy", "COV"),
  ("COVIO2DStrategy", "COV-IO"),
  ("COVOAC2DStrategy", "COV-OAC"),

  ("OAC3DStrategy", "OAC+CBA"),
  ("COVConst3DStrategy", "COV+CBA"),
  ("COVIO3DStrategy", "COV-IO+CBA"),
  ("COVOAC3DStrategy", "COV-OAC+CBA"),

  ## deprecated
  ("COVOAC2Strategy", "COV-OAC "),


) )

# NEW NAMES
pretty_strategy_names_py["LQGStrategy"] = r"(i)LQG$_{\huge\textrm{\emph{no} drag}}$"
pretty_strategy_names_py["iLQGStrategy"] = r"(i)LQG$_{\huge\textrm{drag: baseball}}$"
pretty_strategy_names_py["iLQGSoccerStrategy"] = r"(i)LQG$_{\huge\textrm{drag: soccer}}$"
pretty_strategy_names_py["MPCStrategy"] = r"MPC$_{\huge\textrm{\emph{no} drag}}$"
pretty_strategy_names_py_MPCStrategyStar = r"MPC$^{*}_{\huge\textrm{\emph{no} drag}}$"

# TBC


blues = sns.color_palette("Blues_r")
reds = sns.color_palette("Reds_r")
greens = sns.color_palette("Greens_r")
purples = sns.color_palette("Purples_r")
oranges = sns.color_palette("Oranges_r")

# maps from pretty names to colors
pretty_strategy_colors = {
  #"LQG": reds[0],
  pretty_strategy_names_py["LQGStrategy"]: reds[0],
  "LQR only": reds[1],
  pretty_strategy_names_py["MPCStrategy"]: reds[2],
  pretty_strategy_names_py_MPCStrategyStar: reds[2],
  "MPC-S": reds[3],
  r"MPC {\huge (Unc.)}": reds[4],

  "LQR only (vel=0.01)": oranges[1],
  "LQR only (vel=1)":oranges[2],

  r"LQG {\huge (v=0.01)}": oranges[3],
  r"LQG {\huge (v=1)}": oranges[4],

  "iLQR only": greens[0],
  "iLQR only (soccer)": greens[1],
  pretty_strategy_names_py["iLQGStrategy"]: greens[2],
  pretty_strategy_names_py["iLQGSoccerStrategy"]: greens[3],

  "OAC": blues[0],
  "OAC+CBA": blues[0],
  "OAC + CBA": blues[0],
  "JerkOAC": blues[1],
  "JerkOAC + CBA": blues[1],

  "COV-OAC": purples[0],
  "COV-OAC + CBA": purples[0],
  "COV-OAC+CBA": purples[0],
  "COV-IO": purples[1],
  "COV-IO + CBA": purples[1],
  "COV-IO+CBA": purples[1],
  "COV": purples[2],
  "COV + CBA": purples[2],
  "COV+CBA": purples[2],

}

# pretty_strategy_colors = {
#   "KalmanLQRStrategy": reds[0],
#   "LQRStrategy": reds[1],
#   "MPCStrategy": reds[2],
#   "MPCStrategyHighUncertainty": reds[3],
#
#   "LQRV10Strategy": oranges[1],
#   "KalmanLQRV10Strategy": oranges[2],
#
#   "KalmanLQRV1000Strategy": oranges[3],
#   "LQRV1000Strategy":oranges[4],
#
#   "ELQRStrategy": greens[0],
#   "ELQRSoccerStrategy": greens[1],
#
#   "OACStrategy": blues[0],
#   "COVOAC2Strategy": blues[0],
#   "OAC3DStrategy": blues[0],
#   "JerkOACStrategy": blues[1],
#
#   "OACCOVStrategy": purples[0],
#   "ObsInitCOVStrategy": purples[1],
#   "ConstCOVStrategy": purples[2],
#
# }

pretty_res_types = {
  "distance": "distance",
  "velocity": "velocity",
  "controleffort": "control eff.",
  "J": "J",
}

pretty_strategy_names = None

#=====================================================


def load_experiment_container_to_pandas(experiment_container):
  data = OrderedDict()
  params = OrderedDict()

  for dr_ in os.listdir(experiment_container):
    print "Reading %s" % dr_
    dr = os.path.join(experiment_container, dr_)

    if not os.path.isdir(dr):
      continue

    has_params = False
    for subdr_ in os.listdir(dr):
      subdr = os.path.join(dr, subdr_)

      if os.path.islink(subdr):
        print ("WARN omitting link %s" % subdr)
        continue

      if os.path.isdir(subdr):
        if not has_params:
          with open(os.path.join(subdr, "params.yml"), "r") as f:
            params[dr] =  yaml.load(f)
          has_params = True
        continue

      dist_res = dist_ptn.match(subdr_)
      vel_res = velocity_ptn.match(subdr_)
      ce_res = controleffort_ptn.match(subdr_)

      res = None
      if dist_res is not None:
        res = dist_res
        res_type = "distance"
      if vel_res is not None:
        res = vel_res
        res_type = "velocity"
      if ce_res is not None:
        res = ce_res
        res_type = "controleffort"
        #print "found controleffort"
        #print subdr_

      if res is not None:
        if dr not in data:
          data[dr] = {}

        if res_type + res.group(1) not in data[dr]:
          #print ("?????")
          pass

        if res.group(1) == 'x':
          dt = np.loadtxt(subdr, ndmin=2)
          data[dr][res_type+res.group(1)] = dt[0,:]
        elif res.group(1) == 'z':
          dt = np.loadtxt(subdr, ndmin=2)
          data[dr][res_type+res.group(1)] = dt[:,0]
        elif res.group(1) == 'ymean':
          dt = np.loadtxt(subdr)
          data[dr][res_type+res.group(1)] = dt
        elif res.group(1) == 'y':
          if res.group(2) == "txt":
            print "ignoring %s " % subdr_
            continue
          dt = np.load(subdr)
          data[dr][res_type+res.group(1)] = dt

    for res_type in res_types:
      if res_type+"y" in data[dr] and res_type+"ymean" in data[dr]:
        print "Deleting '%symean'"  % res_type
        del data[dr][res_type+"ymean"]
      else:
        print "WARN: you have an old data format; only means, not all trials are stored"

  data_col_range = range( max ( [ np.prod(data.values()[i]['distancey'].shape) for i in range(len(data)) ] ) )

  data_cols = [ s+str(i) for s in res_types for i in data_col_range ]
  param_cols = sorted(params.values()[0].keys())
  aggr_param_cols = ['strategy']

  strategies = pd.DataFrame(index=data.keys(), columns=data_cols + param_cols + aggr_param_cols)

  for (k, exp_data), (k_, exp_params) in zip(data.items(), params.items()):
    assert (k == k_)
    # TODO these data are only the means; we also have the std but we don't care right now.
    # actually we would like to collect all data points for all runs

    for res_type in res_types:
      try:
        exp_results =  exp_data[res_type+'y'].flatten().tolist()
      except:
        print " ERROR: %s not in %s" % (res_type, k)
        continue
      res_type_data_cols = [ col for col in data_cols if res_type in col ]

      strategies.loc[k][ res_type_data_cols[:len(exp_results)] ] = exp_results
      #strategies.loc[k][exp_params.keys()] = exp_params.values()
      for pk, pv in exp_params.items():
        strategies.loc[k][pk] = pv
      try:
        strategies.loc[k]['strategy'] = strategy_ptn.search(k).group(1)
      except:
        print strategy_ptn_str
        print k
        raise Exception("regex failed")
  strategies[data_cols] = strategies[data_cols].astype(float)

  # 'BallCatching/strategy'
  # 'Ball/drag'
  # 'Agent/noise_std'

  # some clean-up for group-by
  strategies['Ball/wind_gust_force'] = strategies['Ball/wind_gust_force'].apply(str)
  if 'Ball/system_noise_std' in strategies:
    strategies['Ball/system_noise_std'] = strategies['Ball/system_noise_std'].apply(str)
  if 'Ball/observation_noise_std' in strategies:
    strategies['Ball/observation_noise_std'] = strategies['Ball/observation_noise_std'].apply(str)
  if 'Ball/noise_std' in strategies:
    strategies['Ball/noise_std'] = strategies['Ball/noise_std'].apply(str)
  strategies['Agent/noise_std'] = strategies['Agent/noise_std'].apply(str)

  return strategies, data_cols

NOISE_COLUMNS = [
  'Ball/drag',
  'Agent/delay',
  'Ball/wind_gust_force',
  'Ball/observation_noise_std',
  'Ball/system_noise_std',
  'Agent/noise_std',
]


def compute_weighted_score(df, data_cols):
  def J(td, v, ce, w_td=1000, w_v=0., w_ce=0.1):
      return w_td*td**2 + w_v*v**2 + w_ce*ce**2

  matched_cols = zip(*map(sorted, ([ dc for dc in data_cols if "distance" in dc], [ dc for dc in data_cols if "velocity" in dc], [ dc for dc in data_cols if "controleffort" in dc])))

  res_types = ['J']

  param_cols = (df.columns.difference(data_cols)).tolist()
  new_data_cols = [ 'J%d' % i for i in range(len(matched_cols)) ]
  df2 = pd.DataFrame(index=df.index, columns=new_data_cols + param_cols )

  df2[param_cols] = df[param_cols]
  for j, (d, v, c) in zip(new_data_cols, matched_cols):
    df2[j] = J(df[d], df[v], df[c])

  return df2, new_data_cols, res_types

def compute_means(df):
  mean_df = []
  data_col_indexes = []
  for res_type in res_types:
    dcidx = np.where(df.columns.str.contains(res_type))[0]
    data_col_indexes += dcidx.tolist()
    sub = df.icol( dcidx )
    mean_df.append ( sub.mean(skipnna=True, axis=1))
  param_col_indexes = list(  set(range(len(df.columns))) - set(data_col_indexes)  )
  df_wo_vals = df.icol(param_col_indexes)

  merged_df = pd.concat( [df_wo_vals] + mean_df, axis=1)
  merged_df.columns = df_wo_vals.columns.tolist() + res_types

  return merged_df


def df_reorder_by_strategy(groups, res_type):
  res_cols = get_res_type_cols(res_type)
  reorder_dct = {}
  for strat, group in groups.groupby('strategy'):
    v_res = group[res_cols]
    reorder_dct[strat] = np.concatenate([v_res.iloc[i] for i in range(v_res.shape[0])])
  groups_reorder = pd.DataFrame(data=reorder_dct.values(), index=reorder_dct.keys())
  return groups_reorder

def resolve_link(s):
  link_str = "Link to "
  if s.startswith(link_str):
    return s[len(link_str):]
  return s

def make_pretty_strategy_name(s):
  s = resolve_link(s)
  if THREE_D and s == "OACCOVStrategy":
    s += "3D"
  if s not in pretty_strategy_names:
    print ("[make_pretty_strategy_name] WARN: unknown strategy %s" % s)
    pretty_strategy_names[s] = s
    return s
  return pretty_strategy_names[s]

def make_pretty_strategy_color(s):
  if s not in  pretty_strategy_colors.keys():
    print ("[make_pretty_strategy_color] WARN: unknown strategy %s" % s)
    return reds[0]

  #s = resolve_link(s)
  return pretty_strategy_colors[s]

def make_plot(groups_mean, title, label, figsize=None, set_xlim=None):
  if len(groups_mean) == 0:
    print "omitting " + title
    return

  # rename and select colors
  #palette = [ pretty_strategy_colors[pretty_strategy_names[s]] for s in groups_mean.index ]
  groups_mean.index = [ make_pretty_strategy_name(s)  for s in groups_mean.index ]

  if OMIT_LQR:
    groups_mean = groups_mean.iloc[ np.where(np.logical_not(groups_mean.index.str.startswith("LQR")))[0] ]

  # sort_by_distance
  #groups_mean = groups_mean.loc[groups_mean.mean(axis=1).sort(inplace=False).index]
  sorted_idx_and_colors = [ (p, make_pretty_strategy_color(p)) for p in pretty_strategy_names.values() if p in groups_mean.index]
  groups_mean = groups_mean.loc[ [si[0] for si in sorted_idx_and_colors ] ]
  palette = [si[1] for si in sorted_idx_and_colors ]

  # plot
  sns.set(rc=rc_plus)

  #fig = plt.figure() #figsize=(2,1))
  if figsize is not None:
    fig = plt.figure(figsize=figsize)
  else:
    fig = plt.figure()
    print ("Default figure size:", fig.get_size_inches().tolist())
    
  plt.title(title)

  #print sns.axes_style()

 #whis = [1, 99]
  whis = 1e100 # = no outliers

  ax = sns.boxplot(data=groups_mean.T, orient="horizontal", palette=palette, whis=whis,)

  ax.set_xlabel(label)
  if set_xlim is not None:
    ax.set_xlim([0,set_xlim])

  fn = os.path.join(experiment_container, slugify(title)+"."+image_format)
  print "saving %s" % fn
  plt.savefig(fn)

  return ax

## ---------------------------------------------------------------------

experiment_container = None

plot_types = ["ideal",  "worst",
              "drag", "wind", "delay",
              "average",
              "success",
              "gaussian-low", "gaussian-med", "gaussian-high",
              #"drag+gaussian", "wind+gaussian"
              ]
additional_plot_types = [
                         "drag+gaussian", "wind+gaussian", "gaussian",
                         ]

metrics = ["distance", "velocity", "effort",]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("experiment_set", help="experiment set folder")
  parser.add_argument("--no_cache", action="store_true", default=False,  help="whether to load from cache")
  parser.add_argument("--include_lqr", action="store_true", default=False,  help="whether to plot lqr (w/o Kalman), too")
  parser.add_argument("--weighted", action="store_true", default=False,  help="set to true if you want to compute a weighted score (of distance, velocity, ...)")
  parser.add_argument("--three_d", action="store_true", default=False,  help="set to true if evaluating 3D experiment")
  parser.add_argument("--plots", nargs="+", default=plot_types, choices=plot_types+additional_plot_types,
                      help="which plots to generate (leave empty for a default list)")
  parser.add_argument("--strategies", nargs="+", default=None, choices=pretty_strategy_names_py.keys(),
                      help="which strategies to plot (leave empty for all)")
  parser.add_argument("--metrics", nargs="+", default=res_types, choices=res_types,
                      help="which metrics to plot (leave empty for a default list)")
  parser.add_argument("--success_threshold", default=0.5, type=float, help="threshold for binary success plot")
  parser.add_argument("--framerate", default=None, type=int, help="in case you mix framerates, select which one is the 'main' one")
  parser.add_argument("--figsize", nargs="+", type=float, help="figure size (2 only)", default= None)
  parser.add_argument("--noplot", action="store_true", default=False,  help="set to true if no plots wanted")
  parser.add_argument("--mpcstar",  action="store_true", default=False,  help="add a star to MPC (because it's 10 Hz)")
  parser.add_argument("--title",  default=None)

  args = parser.parse_args()

  pretty_strategy_names = pretty_strategy_names_py

  figsize = None
  if args.figsize:
    assert len(args.figsize) == 2
    figsize = args.figsize

  experiment_container = sys.argv[1]

  if not os.path.isdir(experiment_container):
      # try prepending data_root
      experiment_container = os.path.join(data_root, experiment_container)
      assert(os.path.isdir(experiment_container))

  OMIT_LQR = not args.include_lqr
  THREE_D = args.three_d

  res_types = args.metrics

  pkl_strategies_cache = "strategies.pkl"
  pkl_strategies_data_cols_cache = "strategies_data_cols.pkl"

  if not args.no_cache and os.path.exists(os.path.join(experiment_container, pkl_strategies_cache)):
    print "Loading from cache"
    #with open(pkl_strategies_cache, "r") as f:
    try:
      strategies = pd.load(os.path.join(experiment_container, pkl_strategies_cache))
    except:
      # compatibility to older pandas versions
      strategies = pd.read_pickle(os.path.join(experiment_container, pkl_strategies_cache))
    with open(os.path.join(experiment_container, pkl_strategies_data_cols_cache), "rb") as f:
      data_cols = pickle.load(f)

  else:
    strategies, data_cols = load_experiment_container_to_pandas(experiment_container)

    print "Saving"
    strategies.to_pickle(os.path.join(experiment_container, pkl_strategies_cache))
    with open(os.path.join(experiment_container, pkl_strategies_data_cols_cache), "wb") as f:
      pickle.dump(data_cols, f)

  # --------------------------------------------------------------------

  if args.mpcstar:
    pretty_strategy_names["MPCStrategy"] = pretty_strategy_names_py_MPCStrategyStar
    
    # hacky because hard coded framerates as in our experiments
    print ("Adjusting MPC delay to global frame rate")
    assert (args.framerate == 60.)
    fr_nrm = 6
    
    print ("Delays", set(strategies['Agent/delay'].tolist()))
    strategies.loc[:,'Agent/delay'] = strategies['Agent/delay'].map(lambda x: x*fr_nrm if x < 10 else x)
    print ("New delays", set(strategies['Agent/delay'].tolist()))
        

  # --------------------------------------------------------------------
  #strategies.strategy = [ pretty_strategy_names[s]  for s in strategies.strategy ]

  title = ""
  if args.title is not None:
    assert(len(args.metrics) == 1)
    if len(args.plots) > 1:
      warn("len(args.plots) > 1 but custom title")
    title = args.title

  if args.strategies:
    print ("Filtering for strategies")
    strategies = strategies[strategies['strategy'].isin(args.strategies)]

  # compute mean over all trials for each strategy
  # currently unused; boxplot computes this along with the variances automatically
  #strategies_means = compute_means(strategies)

  framerate = set(strategies['BallCatching/framerate'])
  if (len(framerate) != 1):
    if args.framerate is None:
      print framerate
      raise Exception("in case you mix framerates, select which one is the 'main' one by passing --framerate FR")
    else:
      framerate = args.framerate
  else:
    framerate = framerate.pop()


  if args.weighted:
    print "computing weighted"
    strategies, data_cols, res_types = compute_weighted_score(strategies, data_cols)

  # helper function
  def get_res_type_cols(res_type):
    return [ col for col in data_cols if res_type in col]

  has_drag = True in (set(strategies['Ball/drag']))
  # FIXME this is wrong -- we are just checking whether there is more than one noise set (assuming ideal always exists)
  has_wind = len(set(strategies['Ball/wind_gust_force'])) > 1
  has_wind = True
  try:
    has_delay = len(set(strategies['Agent/delay'])) > 1 or list(set(strategies['Agent/delay']))[0] != 0.
  except IndexError:
    warn("WARN: could not extract delay; assuming no delay")
    has_delay = False
  has_gaussian_noise = len(set(strategies['Ball/observation_noise_std'])) > 1 \
    or len(set(strategies['Ball/system_noise_std'])) > 1 \
    or len(set(strategies['Agent/noise_std'])) > 1

  def get_xlim(res_type):
    if res_type == "distance":
      return 5
    if res_type == "J":
      return 5000
    if res_type == "controleffort":
      return 90
    return None

  # --------------------------------------------------------------------
  # prepare delay data
  delay_vals = None
  raw_delay_vals = None

  # existing delay values
  rdv = OrderedDict()
  rdv[0.0] = None
  rdv[0.2] = "200ms"
  rdv[0.4] = "400ms"

  if has_delay:
    delay_vals = sorted(set(strategies['Agent/delay']))
    #assert (len(delay_vals) >= 2)
    #if delay_vals[0] == 0:
    #  del delay_vals[0]  # not needed
    raw_delay_vals = map(lambda x: round(x / float(framerate), 1), delay_vals)

  # --------------------------------------------------------------------
  # IDEAL case
  if "ideal" in args.plots:
    for res_type in res_types:
      if args.title is None:
        title = "Ideal [%s]" % pretty_res_types[res_type]
      groups = strategies[ ['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type) ]
      groups = groups[groups['Ball/drag'] == False]
      groups = groups[groups['Agent/delay'] == 0]
      for col in NOISE_COLUMNS[2:]:
        groups = groups[groups[col] =="[0.0, 0.0, 0.0]"]
      #groups_mean = groups.groupby('strategy').mean()
      groups_mean = df_reorder_by_strategy(groups, res_type)
      make_plot(groups_mean, title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)


  # DRAG only case
  if "drag" in args.plots:
    if has_drag:
      for res_type in res_types:
        if args.title is None:
          title = "Drag [%s]" % pretty_res_types[res_type]
        groups = strategies[['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        groups = groups[groups['Ball/drag'] == True]
        groups = groups[groups['Agent/delay'] == 0]
        for col in NOISE_COLUMNS[2:]:
          groups = groups[groups[col] == "[0.0, 0.0, 0.0]"]
        groups_mean = df_reorder_by_strategy(groups, res_type)
        make_plot(groups_mean, title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)

  # -----------------------------------

  # DELAY only case
  if "delay" in args.plots and delay_vals is not None:
    def make_delay_level(level_name, level_val):
      global title
      # find out which delays there are because it depends on framerate; there should be two, med and high
      for res_type in res_types:
        if args.title is None:
          res_title = "Delay (%s) [%s]" % (level_name, pretty_res_types[res_type])
        else:
          res_title = "%s (%s)" % (title, level_name)
        groups = strategies[['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        groups = groups[groups['Ball/drag'] == False]
        groups = groups[groups['Agent/delay'] == level_val]
        for col in NOISE_COLUMNS[2:]:
          groups = groups[groups[col] == "[0.0, 0.0, 0.0]"]
        groups_mean = df_reorder_by_strategy(groups, res_type)
        make_plot(groups_mean, res_title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)
      return groups

    for draw, d in zip(raw_delay_vals, delay_vals):
      try:
        lbl = rdv[draw]
      except:
        lbl = None
      if lbl is None:
        continue
      groups = make_delay_level(lbl, d)
      #groups = make_delay_level("400ms", delay_vals[1])

  else:
    if "delay" in args.plots:
      print ("omitting delay - no experiment found")


  # WORST case
  if "worst" in args.plots:
    if has_gaussian_noise:

      def make_worst_case_level(level_idx):
        global title
        
        # idx \in [0,1,2]

        std_idx = [ "0.001", "0.01", "0.1" ]
        delay_idx = delay_vals
        title_idx = ["low", "med.", "high"]

        print ("----")
        print ("Disturbances: %s" % title_idx[level_idx])

        all_groups = []

        for res_type in res_types:
          if args.title is None:
            res_title = "Combined (%s) [%s]" % (title_idx[level_idx], pretty_res_types[res_type])
          else:
            res_title = "%s" % title
          groups = strategies[ ['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type) ]
          #groups = groups[groups['Ball/drag'] == True] # we want worst case with and without drag

          # DELAY
          if level_idx == 0:
            print ("  -> no delay" )
            groups_ = groups[groups['Agent/delay'] == 0]

            if len(groups_) == 0:
              print ("WARN: Worst-case is WRONG because DELAY=0 is not available")
              res_title += " *with DELAY?!"
            else:
              groups = groups_

          else:
            if delay_vals is None:
              print ("WARN: Worst-case is not worst because DELAY is not set")
              res_title += " *w/o delay"
            else:
              rdl_des = rdv.keys()[level_idx]
              dl_des = int (rdl_des*framerate)
              print ("delay vals: " + str(dl_des))
              groups_ = groups[groups['Agent/delay'] == dl_des]
              if len(groups_) == 0:
                print ("WARN: Worst-case is WRONG because DELAY=%d is not available" % dl_des)
                res_title += " *with DELAY?!"
              else:
                groups = groups_

          # DRAG
          # nothing - we average over ideal and drag to make comparison fair
          pass

          # WIND
          if level_idx == 0:
            print ("  -> no wind")
            pass
          else:
            groups_ = groups[groups["Ball/wind_gust_force"] != "[0.0, 0.0, 0.0]"]
            if len(groups_) == 0:
              print ("WARN: Worst-case is not worst because WIND is not set")
              res_title += " *w/o wind"
            else:
              groups = groups_

          # GAUSSIAN
          for col in NOISE_COLUMNS[3:]:
            groups_ = groups[groups[col] !="[0.0, 0.0, 0.0]"]
            if len(groups_) == 0:
              print ("WARN: Worst-case is not worst because %s is not set" % col)
              res_title += " *w/o %s" % col
            else:
              groups = groups_
          # ---

          # FIXME
          # if THREE_D:
          #   level_std = "[0.1, 0.1, 0.1]"
          # else:
          #   level_std = "[0.1, 0.1, 0.0]"

          
          # try 2d
          level_std = "[%s, %s, 0.0]" % tuple([std_idx[level_idx]] * 2)
      
          groups_ = groups[groups['Agent/noise_std'] == level_std]
          # these two lines are implicit because we do not evaluate gaussian noise level combinations
          groups_ = groups_[groups['Ball/observation_noise_std'] == level_std]
          groups_ = groups_[groups['Ball/system_noise_std'] == level_std]
          
          if len(groups_) == 0:
            level_std = "[%s, %s, %s]" % tuple([std_idx[level_idx]] *3 )
            groups_ = groups[groups['Agent/noise_std'] == level_std]
            # these two lines are implicit because we do not evaluate gaussian noise level combinations
            groups_ = groups_[groups['Ball/observation_noise_std'] == level_std]
            groups_ = groups_[groups['Ball/system_noise_std'] == level_std]
            
            if len(groups_) > 0:
              print ("3D setting!")
          
          print ("  gaussian level std: " + level_std)
          groups = groups_          
          
          # ---
          #groups_mean = groups.groupby('strategy').mean()
          groups_mean = df_reorder_by_strategy(groups, res_type)
          make_plot(groups_mean, res_title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)

          all_groups.append(groups)
          
        return all_groups

      groups = make_worst_case_level(0)
      groups = make_worst_case_level(1)
      groups = make_worst_case_level(2)

  # -----------------------------------
  # AVERAGE
  if "average" in args.plots:
    if has_gaussian_noise or has_drag or has_wind or has_delay:
      for res_type in res_types:
        if args.title is None:
          title = r"\textbf{Average} [%s]" % pretty_res_types[res_type]
        groups = strategies[ ['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        groups_mean = df_reorder_by_strategy(groups, res_type)
        make_plot(groups_mean, title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)

  # # -----------------------------------
  # # SUCCESS RATE
  if "success" in args.plots and "distance" in res_types:
    if has_gaussian_noise or has_drag or has_wind or has_delay:
      for res_type in res_types:
        if res_type != "distance":
          continue

        if args.title is None:
          title = r"\textbf{Average success rate} ($< %.1f$)" % (args.success_threshold,)
        groups = strategies[['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]

        # if False:
        #   # FIXME ideal success
        #   groups = strategies[['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        #   groups = groups[groups['Ball/drag'] == False]
        #   groups = groups[groups['Agent/delay'] == 0]
        #   for col in NOISE_COLUMNS[2:]:
        #     groups = groups[groups[col] == "[0.0, 0.0, 0.0]"]

        groups2 = groups.copy()
        groups2.loc[:, get_res_type_cols(res_type)] = groups.loc[:, get_res_type_cols(res_type)] < args.success_threshold
        groups2.loc[:, get_res_type_cols(res_type)] = groups2.loc[:, get_res_type_cols(res_type)].apply(lambda y: map(lambda x: int(x), y))
        # insert NaNs back because the comparison has removed them
        groups2[groups.isnull()] = np.nan

        #print groups2
        print "---------------"
        print title
        groups_mean = df_reorder_by_strategy(groups2, res_type)
        precision = groups_mean.mean(axis=1)
        precision.sort_values(inplace=True, ascending=False)
        precision.index = [make_pretty_strategy_name(s) for s in precision.index]
        print precision

        fn = os.path.join(experiment_container, slugify(title) + ".tex")
        with open(fn, "w") as f:
          precision.to_frame().to_latex(header=False, formatters=[lambda x: "%.2f" % x], buf=f)

        #print groups_mean.std(axis=1)
        # plot is pretty useless
        #make_plot(groups_mean, title, pretty_res_types[res_type], set_xlim=1, figsize=figsize)


  # -----------------------------------
  #  Ideal and Gaussian disturbances
  if has_gaussian_noise:
    if "gaussian" in args.plots:
      for res_type in res_types:
        # TODO actually ideal is in here too - don't care?
        if args.title is None:
          #title = "Gaussian disturbances [%s]" % pretty_res_types[res_type]
          title = "Gaussian [%s]" % pretty_res_types[res_type]
        groups = strategies[ ['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        groups = groups[groups['Ball/drag'] == False]
        groups = groups[groups['Agent/delay'] == 0]
        groups = groups[groups['Ball/wind_gust_force'] =="[0.0, 0.0, 0.0]"]
        groups_mean = df_reorder_by_strategy(groups, res_type)
        make_plot(groups_mean, title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)

    def make_gauss_level(level_name, level_std):
      global title
      
      for res_type in res_types:
        # TODO actually ideal is in here too - don't care?
        if args.title is None:
          #title = "Gaussian disturbances (%s) [%s]" % (level_name, pretty_res_types[res_type])
          res_title = "Gaussian (%s) [%s]" % (level_name, pretty_res_types[res_type])
        else:
          res_title = "%s (%s)" % (title, level_name)
        groups = strategies[ ['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        groups = groups[groups['Ball/drag'] == False]
        groups = groups[groups['Agent/delay'] == 0]
        groups = groups[groups['Ball/wind_gust_force'] =="[0.0, 0.0, 0.0]"]
        #groups = groups[groups['Agent/noise_std'] == level_std]
        groups = groups[groups['Ball/observation_noise_std'] == level_std]
        #groups = groups[groups['Ball/system_noise_std'] == level_std]
        groups_mean = df_reorder_by_strategy(groups, res_type)
        make_plot(groups_mean, res_title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)
      return groups

    if "gaussian-low" in args.plots:
      lvl = "[0.001, 0.001, 0.0]"
      groups= make_gauss_level("low", lvl)
      if len(groups) == 0:
        lvl = "[0.001, 0.001, 0.001]"
        groups= make_gauss_level("low", lvl)
 
    if "gaussian-med" in args.plots:
      lvl = "[0.01, 0.01, 0.0]"
      groups= make_gauss_level("med", lvl)
      if len(groups) == 0:
        lvl = "[0.01, 0.01, 0.01]"
        groups= make_gauss_level("med", lvl)

    if "gaussian-high" in args.plots:
      lvl = "[0.1, 0.1, 0.0]"
      groups= make_gauss_level("high", lvl)
      if len(groups) == 0:
        lvl = "[0.1, 0.1, 0.1]"
        groups= make_gauss_level("high", lvl)

  # -----------------------------------
  #  DRAG and Gaussian only   (@deprecated)
  if "drag+gaussian" in args.plots:
    if has_gaussian_noise and has_drag:
      for res_type in res_types:
        if args.title is None:
          title = ("Drag and Gaussian disturbances [%s]" % pretty_res_types[res_type])
        groups = strategies[ ['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        groups = groups[groups['Ball/drag'] == True]
        groups = groups[groups['Agent/delay'] == 0]
        groups = groups[groups['Ball/wind_gust_force'] =="[0.0, 0.0, 0.0]"]
        groups_mean = df_reorder_by_strategy(groups, res_type)
        make_plot(groups_mean, title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)

  # -----------------------------------
  if "wind" in args.plots:
    if has_wind:
      for res_type in res_types:
        #  WIND only
        if args.title is None:
          title = ("Wind gust [%s]" % pretty_res_types[res_type])
        groups = strategies[ ['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        groups = groups[groups['Ball/drag'] == False]
        groups = groups[groups['Agent/delay'] == 0]
        groups = groups[groups['Ball/wind_gust_force'] !="[0.0, 0.0, 0.0]"]
        for col in NOISE_COLUMNS[3:]:
          groups = groups[groups[col] =="[0.0, 0.0, 0.0]"]
        groups_mean = df_reorder_by_strategy(groups, res_type)
        make_plot(groups_mean, title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)

  if "wind+gaussian" in args.plots:
    if has_gaussian_noise:
      for res_type in res_types:
        #  WIND and gaussian only
        if args.title is None:
          title = ("Wind and Gaussian disturbances [%s]" % pretty_res_types[res_type])
        groups = strategies[ ['strategy'] + NOISE_COLUMNS + get_res_type_cols(res_type)]
        groups = groups[groups['Ball/drag'] == False]
        groups = groups[groups['Ball/wind_gust_force'] !="[0.0, 0.0, 0.0]"]
        groups = groups[groups['Agent/delay'] == 0]
        groups_mean = df_reorder_by_strategy(groups, res_type)
        make_plot(groups_mean, title, pretty_res_types[res_type], set_xlim=get_xlim(res_type), figsize=figsize)

  # -----------------------------------

  if not args.noplot:  
    plt.show()
    
  sys.exit()
