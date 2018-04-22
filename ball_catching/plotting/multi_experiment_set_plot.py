#!/usr/bin/python

###############################################
# Plotting: Multiple experiments

from ball_catching.plotting.multi_experiment_plot import *
from ball_catching.utils.utils import load_experiments
import glob

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Usage: set_multi_experiment_plot.py [--no_plot] <experiment_container> [<varying param1>, <varying param2>, ...]"

  do_plot = True
  if sys.argv[1].strip() == "--no_plot":
    print "no plot"
    do_plot = False
    del sys.argv[1]

  experiment_container = sys.argv[1]
  
  for experiment_name in os.listdir(os.path.join(data_root,experiment_container)):
    f = os.path.join(data_root,experiment_container,experiment_name)
    if not os.path.isdir(f):
      continue
    
    print "-------------"

    # FIXME: check if pdf has already been generated
    if len(glob.glob("*.pdf")) > 0:
        print " omitting %s"
        continue
 
    print "Plotting %s " % experiment_name
    
    experiments = load_experiments(os.path.join(experiment_container,experiment_name))

    # assume varying params stays constant
    X,Z,labels, AVG = collect_varying_parameters(experiments, sys.argv[2:])

    plot_agent_ball_distance(experiment_name, experiments, X, Z, labels, AVG, plot=do_plot)
    plot_catch_velocity_and_control_effort(experiment_name, experiments, X, Z, labels, AVG, plot=do_plot)

    #plot_cov_rv(experiment_name, experiments, X, Z, labels, AVG)

    #plot_catch_interpolation_error(experiment_name, experiments, X, Z, labels, AVG)
    
    #plot_ball_impact_point(experiment_name, experiments, X, Z, labels, AVG)
    
    #generate_drag_data_file(experiment_name, experiments, X, Z, labels, AVG)

    if do_plot:
      plt.show()