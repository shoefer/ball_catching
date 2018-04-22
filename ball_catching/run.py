import sys
import os
import argparse
import json
import copy
import shutil
import shelve

from ball_catching.dynamics.main import run_python
from ball_catching.settings import EXPERIMENT_RANGES_2D, EXPERIMENT_RANGES_3D, NOISE_TYPES, STATS
from ball_catching.strategies import STRATEGIES
from ball_catching.experiment import single_experiment, multi_experiment
from ball_catching.utils.utils import *
from ball_catching import dictionaries

###############################################
# Initialize strategies

for s in STRATEGIES:
  if s not in dictionaries:
      # create an empty dictionary for each strategy
      create_dictionary(s, {})

###############################################
# Run experiments

if __name__ == "__main__":

    # Log the command that was executed
    _cmd = " ".join(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str.lower, choices=["multi", "multi3d",
                                                         "single", "single3d",
                                                         "const-cov-adversarial",
                                                         ],
                        default="", help="experiment type")
    parser.add_argument("--trials", type=int, default=1, help="number of trials per run")
    parser.add_argument("--noise", type=str.lower, default=None,
                        choices=NOISE_TYPES.keys() + ['all', 'full_comb', 'full_gaussian', 'full_gaussian_drag',
                                                      'full_gaussian_delay'],
                        help="required for single and optional for multi - which noise to simulate (in multi leave unset for no noise, use 'all' for using all noise sources, and 'full_comb' the combination of noise sources)",
                        nargs="+")
    # parser.add_argument("--noise_high_only", action="store_true", default=False, help="only use high noise") # TODO
    parser.add_argument("--noise_medium_only", action="store_true", default=False, help="only use medium noise")
    parser.add_argument("--noise_low_only", action="store_true", default=False, help="only use low noise")
    parser.add_argument("--noise_high_only", action="store_true", default=False, help="only use high noise")
    parser.add_argument("--noise_only", action="store_true", default=False,
                        help="do not do ideal (need to set some 'noise' source)")  #
    parser.add_argument("--strategies", type=str,
                        choices=STRATEGIES, default=[], nargs="+",
                        help="set the strategies that should be evaluated")
    parser.add_argument("--range", choices=EXPERIMENT_RANGES_2D.keys() + EXPERIMENT_RANGES_3D.keys(), type=str.lower,
                        default="m", help="configuration range")
    parser.add_argument("--params", type=json.loads, default={}, help="override the parameters with a dictionary")
    parser.add_argument("--params_yml", type=str, default=None, help="override the parameters with a yml file")
    parser.add_argument("--framerate", default=None, help="Override control framerate", type=int)
    # parser.add_argument("--no_plot", action="store_true", default=False, help="Disable plotting")
    parser.add_argument("--no_plot", action="store_true", default=False, help="Don't plot")

    parser.add_argument("--data_root", default=None, help="Alternative data_root")
    args = parser.parse_args()

    trials = args.trials

    dictionaries['BallCatching']['_backend'] = "python"

    if args.framerate is not None:
        print("Adjusting framerate=%d" % args.framerate)
        dictionaries['BallCatching']['framerate'] = args.framerate

    if args.data_root is not None:
        data_root = args.data_root
        print("Alternative data root: %s" % data_root)

    # change noise levels
    if args.noise_medium_only:
        set_noise_levels('med')
    if args.noise_low_only:
        set_noise_levels('low')
    if args.noise_high_only:
        set_noise_levels('high')

    if len(args.params) > 0:
        print("------")
        # Update global dictionary with parameter overrides
        print("Overriding global parameters settings")
        new_params = make_hierarchical(args.params)
        update_dictionary(new_params, verbose=True)
        print("------")

    if args.params_yml is not None:
        print("Overriding global parameters settings")
        new_params = load_yaml(args.params_yml)
        update_dictionary(new_params, verbose=True)
        print("------")

    # ------------------------------------
    # prepare experiment sets

    if args.type == "single" or args.type == "single3d":
        if len(args.strategies) != 1:
            raise Exception(
                "single/single3d: You need to provide exactly one strategy! (not %d)" % len(args.strategies))
        experiment_container_root, experiment_sets = single_experiment(
            args.strategies[0], trials=args.trials, noise_fixed=args.noise,
            three_d=args.type == "single3d")

    elif args.type == "multi" or args.type == "multi3d":
        experiment_container_root, experiment_sets = multi_experiment(args.strategies,
                                                                      ball_range=args.range, trials=args.trials,
                                                                      noise_fixed=args.noise,
                                                                      three_d=args.type == "multi3d",
                                                                      noise_only=args.noise_only)


    elif args.type == "const-cov-adversarial":
        delete_experiment_folders = False

        strategy_label = "COVConst2DStrategy"
        strategy_name = "COVOAC2DStrategy"
        rv_delay = -1

        parameters = {
            strategy_name: {}
        }
        # COV: adversarial COV strategy case for V=30, a0=10 (agent runs away from goal)
        parameters[strategy_name]['rv_averaging'] = 1.
        parameters[strategy_name]['rv_delay'] = rv_delay

        # python
        parameters[strategy_name]['rv_const'] = parameters[strategy_name]['rv']
        parameters[strategy_name]['rv_averaging'] = 1
        parameters[strategy_name]['dtheta_averaging'] = 0
        parameters[strategy_name]['ddtheta_min'] = 0

        experiment_container_root, experiment_sets = single_experiment(strategy_label,
                                                                       parameters,
                                                                       trials=args.trials,
                                                                       ax=10, V=30,
                                                                       noise_fixed=None)

    else:
        print
        "Experiment type not supported! type: %s" % args.type

    print
    "Experiment sets (#%d):\n   %s" % (len(experiment_sets), "\n   ".join(experiment_sets.keys()))

    # ------------------------------------
    # print some info
    no_runs = sum([len(x) for x in experiment_sets.itervalues()])
    print
    "TOTAL number of runs (parametrizations) %d" % no_runs

    if no_runs > 1000:
        print
        "Starting in "
        from time import sleep

        for i in reversed(range(3)):
            print
            " %d..." % (i + 1)
            sleep(1)
        print
        "GO!"


    # ------------------------------------
    i = 0

    for experiment_name, parametrizations in experiment_sets.iteritems():
        experiment_root = os.path.join(data_root, experiment_name)
        if experiment_container_root != "":
            experiment_root = os.path.join(data_root, experiment_container_root, experiment_name)

        if os.path.exists(experiment_root):
            raise Exception("Experiment root path %s already exists!" % experiment_root)

        os.makedirs(experiment_root)

        # writing cmd
        cmd_folder = os.path.join(data_root,
                                  experiment_container_root) if experiment_container_root != "" else experiment_root
        with open(os.path.join(cmd_folder, "cmd.txt"), "w") as f:
            f.write(_cmd)

        print
        "############################################################"
        print ("Running %s" % experiment_name)

        for parametrization in parametrizations:

            print ("############################################################")
            print ("Run %d/%d" % (i + 1, no_runs))
            i += 1

            # update parametrization
            update_dictionary(parametrization, verbose=True)

            # create folder
            log_folder = "BC_%s_%s" % (dictionaries['BallCatching']['strategy'], generate_timestamp())
            log_root = os.path.join(experiment_root, log_folder)

            # check that folder does not exist and create
            if os.path.exists(log_root):
                raise Exception("Log root path %s already exists!" % log_root)
            os.makedirs(log_root)
            print
            "Experiment log folder is: %s" % log_root

            dicts = merge_dictionaries()
            write_parameter_file(log_root, dicts)

            # create yaml and load to ros parameter server
            t, X, U, X_noisy, strategy_obj = run_python(dicts, log_root)
            # stats = [ np.linalg.norm(X[-1,[0,6]] - X[-1,[9,12]]) ]
            e = Experiment(log_root, log_folder)
            stats = get_stats_for_experiment(e)

            # compress_log files in case you are running a lot of experiments
            print ("Statistics: (trials %d)" % dicts['BallCatching/trials'])
            for l, s in zip(STATS, stats):
                print("%s -> %s" % (l, str(s)))

    ###############################################
    # Plotting: Single experiment
    if not args.no_plot:  # and rospy is not None:

        if len(experiment_sets) == 1 and len(experiment_sets.values()[0]) == 1:
            from ball_catching.plotting.single_experiment_plot import plot_single_experiment
            plot_single_experiment(log_root, dictionaries, experiment_sets, experiment_root)
        else:
            if len(experiment_sets) <= 3:
                from ball_catching.plotting.multi_experiment_plot import multi_experiment_plot

                experiment_container_path = os.path.join(data_root, experiment_container_root)
                for k in experiment_sets.keys():
                    try:
                        multi_experiment_plot(os.path.join(experiment_container_path, k))
                    except Exception as e:
                        print(e)
            else:
                print
                "WARN: Cannot plot because we have too many multi strategies (> 3)"