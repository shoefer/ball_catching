from ball_catching.utils.utils import *

from ball_catching.settings import EXPERIMENT_RANGES_2D, EXPERIMENT_RANGES_3D, NOISE_PARAMS_OFF, NOISE_LEVELS, NOISE_TYPES

def compute_delay_steps(delay_ms, framerate):
    return int(round(delay_ms * framerate))


def set_noise_levels(level):
    # NOISE_LEVELS ={'low': [0.001]*3, 'med': [0.01]*3, 'high': [0.1]*3}
    global NOISE_LEVELS, NOISE_TYPES

    NOISE_LEVELS = {level: NOISE_LEVELS[level]}
    if level != 'low':
        # NOISE_TYPES[level] = {level: NOISE_TYPES["delay"][level]}
        NOISE_TYPES["delay"] = {level: NOISE_TYPES["delay"][level]}
    else:
        NOISE_TYPES["delay"] = {}

    # if level=='high':
    #   NOISE_LEVELS =  {'high': NOISE_LEVELS["high"]}
    #   NOISE_TYPES["high"] = {'high': NOISE_TYPES["delay"]["high"]}
    # elif level == 'med':
    #   NOISE_LEVELS = {'med': [0.01] * 3, }
    #   NOISE_TYPES["delay"] = {'med': NOISE_TYPES["delay"]["med"]}
    # elif level=='low':
    #   NOISE_LEVELS ={'low': [0.001]*3,}
    #   NOISE_TYPES["delay"] = {'low': NOISE_TYPES["delay"]["low"]}

    NOISE_TYPES["agent"] = NOISE_LEVELS
    NOISE_TYPES["observation"] = NOISE_LEVELS
    NOISE_TYPES["system"] = NOISE_LEVELS

    print ("Adapted noise levels: ")
    print (NOISE_LEVELS)

def generate_experiment_sets(strategy, experiment_type, parametrizations, experiment_sets, expose_params=[]):
    # ELQR specific thing
    experiment_set_parametrizations = []

    # if strategy == "ELQRStrategy":
    # import copy
    # for gain_folder in ['elqr_far']:#['elqr_close', 'elqr_med', 'elqr_far']:
    # new_params = []
    # for p_ in parametrizations:
    # p = copy.deepcopy(p_)
    # p['ELQRStrategy'] = p['ELQRStrategy'] if 'ELQRStrategy' in p else {}
    # p['ELQRStrategy']['gain_folder'] = gain_folder
    # new_params.append(p)

    # experiment_set_parametrizations.append( new_params )
    # else:
    # experiment_set_parametrizations.append(parametrizations)

    experiment_set_parametrizations.append(parametrizations)

    # -------------------
    # collect parametrizations and create a name for the experiment

    for parametrizations in experiment_set_parametrizations:

        # get name for type
        has_projected_angle = (strategy == "OACStrategy" or strategy == "OACConstStrategy") and dictionaries[strategy][
            'use_projected_angle']

        has_observation_noise = np.max(parametrizations[0]['Ball']['observation_noise_std']) > 0
        has_system_noise = np.max(parametrizations[0]['Ball']['system_noise_std']) > 0
        has_agent_noise = np.any(np.array(parametrizations[0]['Agent']['noise_std']) > 0)
        has_drag = parametrizations[0]['Ball']['drag'] == True

        has_wind = max(map(np.abs, parametrizations[0]['Ball']['wind_gust_force'])) > 0.
        has_pos_wind, has_neg_wind = False, False
        if has_wind:
            has_neg_wind = min(parametrizations[0]['Ball']['wind_gust_force']) < 0.
            has_pos_wind = not has_neg_wind
            # print "WIND"
            # print has_wind, has_pos_wind, has_neg_wind

        _delay = parametrizations[0]['Agent']['delay']
        has_delay = _delay > 0
        type = []

        # if strategy=="COVStrategy" :
        if strategy[-len("COVStrategy"):] == "COVStrategy":
            real_strategy = "COVStrategy"
            if parametrizations[0][real_strategy]['rv_mode'] == "constant":
                if strategy == real_strategy:
                    strategy = "Const" + real_strategy  # only locally
            elif parametrizations[0][real_strategy]['rv_mode'] == "observe_init":
                # type.append("rvAvg%d" %  parametrizations[0][real_strategy]['rv_averaging'])
                # type.append("rvDel%d" %  parametrizations[0][real_strategy]['rv_delay'])
                # type.append("tadAvg%d" %  parametrizations[0][real_strategy]['tan_alpha_dot_averaging'])
                if strategy == real_strategy:
                    strategy = "ObsInit" + strategy  # only locally
            elif parametrizations[0][real_strategy]['rv_mode'] == "continuous_update":
                # type.append("rvAvg%d" %  parametrizations[0][real_strategy]['rv_averaging'])
                # type.append("rvDel%d" %  parametrizations[0][real_strategy]['rv_delay'])
                # type.append("tadAvg%d" %  parametrizations[0][real_strategy]['tan_alpha_dot_averaging'])
                if strategy == real_strategy:
                    strategy = "OAC" + strategy  # only locally

        # if strategy=="OACStrategy":
        if strategy[-len("OACStrategy"):] == "OACStrategy":
            real_strategy = "OACStrategy"
            # deprecated
            if parametrizations[0][real_strategy]['c'] < 1.:
                type.append("Damped%f" % dictionaries[strategy]['c'])
            if parametrizations[0][real_strategy]['jerk_control'] and strategy != "JerkOACStrategy":
                strategy = "Jerk" + strategy  # only locally

        if strategy[-len("OACRawStrategy"):] == "OACRawStrategy" or \
                        strategy[-len("COVOAC2DRawStrategy"):] == "COVOAC2DRawStrategy":
            real_strategy = strategy
            weight_ptn = re.compile(r"W\_[0-9]+")
            has_weight = len([p for p in parametrizations[0][real_strategy] if weight_ptn.match(p)]) > 0
            uses_linear = parametrizations[0][real_strategy]['W_linear']
            if has_weight and uses_linear:
                strategy += "CustomWeights"
            else:
                clf = parametrizations[0][real_strategy]['clf']
                strategy += os.path.basename(clf).split("__")[0].upper()
            if parametrizations[0][real_strategy]['control'] == "alpha":
                strategy += "-ALPHA"
            elif parametrizations[0][real_strategy]['control'] == "theta":
                strategy += "-THETA"
            elif parametrizations[0][real_strategy]['control'] == "dtheta":
                strategy += "-DTHETA"

        if strategy == "LQRStrategy":
            if parametrizations[0][strategy]['open_loop'] == True:
                type.append("OpenLoop")
            if parametrizations[0][strategy]['time_approx'] == True:
                type.append("TimeApprox")
            if parametrizations[0][strategy]['use_kalman_filter']:
                type.append("Kalman")
            if "V10" in dictionaries[strategy]['gain_folder']:
                type.append("TerminalVel")
            if "V1000" in dictionaries[strategy]['gain_folder']:
                type.append("V1000")

        if strategy == "ELQRStrategy":
            # print dictionaries[strategy]['gain_folder'][-6:]
            if dictionaries[strategy]['gain_folder'][-6:] == "soccer":
                type.append("Soccer")

            try:
                type.append(parametrizations[0][strategy]['gain_folder'])
            except:
                print
                " ELQRStrategy: using default for naming experiment set"
                type.append(dictionaries[strategy]['gain_folder'])

        if has_projected_angle:
            type.append("projAngle")
        if has_observation_noise:
            s = "observationNoise"
            if np.max(parametrizations[0]['Ball']['observation_noise_std']) < 0.1:
                if np.max(parametrizations[0]['Ball']['observation_noise_std']) < 0.01:
                    s += "Low"
                else:
                    s += "Med"
            else:
                s += "High"
            type.append(s)
        if has_system_noise:
            s = "systemNoise"
            if np.max(parametrizations[0]['Ball']['system_noise_std']) < 0.1:
                if np.max(parametrizations[0]['Ball']['system_noise_std']) < 0.01:
                    s += "Low"
                else:
                    s += "Med"
            else:
                s += "High"
            type.append(s)
        if has_agent_noise:
            s = "agentNoise"
            if np.max(parametrizations[0]['Agent']['noise_std']) < 0.1:
                if np.max(parametrizations[0]['Agent']['noise_std']) < 0.01:
                    s += "Low"
                else:
                    s += "Med"
            else:
                s += "High"
            type.append(s)
        if has_drag:
            type.append("drag")
        if has_wind:
            if has_pos_wind:
                type.append("windPos")
            elif has_neg_wind:
                type.append("windNeg")
            else:
                raise Exception("What kind of wind?!")

        if has_delay:
            for k, v in NOISE_TYPES["delay"].items():
                if _delay == v or _delay == compute_delay_steps(v, dictionaries['BallCatching']['framerate']):
                    type.append("delay" + k[0].upper() + k[1:])
                    break

                    # if len(expose_params) > 0:
                    ##ep_str = "_".join([ pcat+"-"+pname + "-" +str(parametrizations[0][pcat][pname]) for pcat, pname in expose_params])
                    # ep_str = "_".join([ pname + "-" +str(parametrizations[0][pcat][pname]) for pcat, pname in expose_params])
                    # type.append("_" +ep_str)

        # finish
        if len(type) == 0:
            type = "ideal"
        else:
            type = "_".join(type)

        # add framerate
        fr_str = "DTinv-%d" % dictionaries['BallCatching']['framerate']

        experiment_name = "%s_%s_%s_%s_%s" % (strategy, experiment_type, type, fr_str, generate_timestamp())

        experiment_sets[experiment_name] = parametrizations

    return experiment_sets


# -----------------------------------------------------------------------
def resolve_strategy(strategy, p):
    # strategy renaming
    real_strategy_name = strategy
    if strategy == "ObsInitCOVStrategy":
        real_strategy_name = "COVStrategy"
        p['COVStrategy'] = dictionaries['COVStrategy'].copy()
        p['COVStrategy']['rv_mode'] = "observe_init"
    elif strategy == "ConstCOVStrategy":
        real_strategy_name = "COVStrategy"
        p['COVStrategy'] = dictionaries['COVStrategy'].copy()
        p['COVStrategy']['rv_mode'] = "constant"
    elif strategy == "OACCOVStrategy":
        real_strategy_name = "COVStrategy"
        p['COVStrategy'] = dictionaries['COVStrategy'].copy()
        p['COVStrategy']['rv_mode'] = "continuous_update"

    elif strategy == "JerkOACStrategy":
        real_strategy_name = "OACStrategy"
        # jerk settings
        p['OACStrategy'] = dictionaries['OACStrategy'].copy()
        p['OACStrategy']['threshold'] = 0.03
        p['OACStrategy']['jerk_control'] = True
    elif strategy == "OACStrategy":
        # non-jerk settings
        real_strategy_name = "OACStrategy"
        p['OACStrategy'] = dictionaries['OACStrategy'].copy()
        p['OACStrategy']['threshold'] = 0.
        p['OACStrategy']['jerk_control'] = False

    elif strategy == "KalmanLQRStrategy":
        # with kalman filtering
        real_strategy_name = "LQRStrategy"
        p['LQRStrategy'] = dictionaries['LQRStrategy'].copy()
        p['LQRStrategy']['use_kalman_filter'] = True

    ###
    elif strategy == "KalmanLQRV10Strategy":
        # with kalman filtering and terminal velocity constraint
        real_strategy_name = "LQRStrategy"
        p['LQRStrategy'] = dictionaries['LQRStrategy'].copy()
        p['LQRStrategy']['use_kalman_filter'] = True
        p['LQRStrategy']['gain_folder'] = "lqrV10"

    elif strategy == "LQRV10Strategy":
        # with terminal velocity constraint
        real_strategy_name = "LQRStrategy"
        p['LQRStrategy'] = dictionaries['LQRStrategy'].copy()
        p['LQRStrategy']['gain_folder'] = "lqrV10"

    ###
    elif strategy == "KalmanLQRV1000Strategy":
        # with kalman filtering and terminal velocity constraint
        real_strategy_name = "LQRStrategy"
        p['LQRStrategy'] = dictionaries['LQRStrategy'].copy()
        p['LQRStrategy']['use_kalman_filter'] = True
        p['LQRStrategy']['gain_folder'] = "lqrV1000"

    elif strategy == "LQRV1000Strategy":
        # with terminal velocity constraint
        real_strategy_name = "LQRStrategy"
        p['LQRStrategy'] = dictionaries['LQRStrategy'].copy()
        p['LQRStrategy']['gain_folder'] = "lqrV1000"

    ###
    elif strategy == "ELQRSoccerStrategy":
        # with kalman filtering
        real_strategy_name = "ELQRStrategy"
        p['ELQRStrategy'] = dictionaries['ELQRStrategy'].copy()
        p['ELQRStrategy']['gain_folder'] += "_soccer"

    else:
        p[strategy] = dictionaries[strategy].copy()

    # print "  Resolved %s to %s"  % (strategy, real_strategy_name)

    return real_strategy_name, p


def initialize_params_3d(p, D, aphi=np.pi / 4):
    # print "3D experiment; aphi = " + str(aphi) + ", D=" + str(D)
    p['Agent']['x_init_type'] = 'spherical'
    p['Agent']['x_relative'] = True
    p['Agent']['x_init'] = [float(aphi), 0., float(D)]
    if 'BallCatching' not in p:
        p['BallCatching'] = {}
    p['BallCatching']['dim'] = 3

    return p


# =======================================================================
# default experiment
def single_experiment(strategy, parameters={}, trials=1,
                      ax=10, V=30,
                      # ax=2.5, V=15,
                      noise_fixed=None, three_d=False,
                      theta=np.pi / 4, aphi=np.pi / 4, drag=False):
    global dictionaries

    experiment_sets = {}
    # parametrizations = []
    p = {}
    p['BallCatching'] = {
        'strategy': dictionaries['BallCatching']['strategy'],
        'dim': 3 if three_d else 3,  # dictionaries['BallCatching']['dim'],
    }
    p['Ball'] = {}
    p['Agent'] = {}
    p['Agent'] = {'noise_std': dictionaries['Agent']['noise_std'],
                  'delay': dictionaries['Agent']['delay']}
    p['Ball'] = {'v_init_type': 'spherical',
                 'observation_noise_std': dictionaries['Ball']['observation_noise_std'],
                 'system_noise_std': dictionaries['Ball']['system_noise_std'],
                 'noise_dist_factor': dictionaries['Ball']['noise_dist_factor'],
                 'drag': dictionaries['Ball']['drag'],
                 'wind_gust_force': dictionaries['Ball']['wind_gust_force']
                 }

    # --------------------------
    # TODO noise_fixed

    # ideal:
    p['Ball']['observation_noise_std'] = [0., 0., 0.]
    p['Ball']['system_noise_std'] = [0., 0., 0.]
    p['Agent']['noise_std'] = [0., 0., 0.]
    p['Ball']['wind_gust_force'] = [0., 0., 0.]
    p['Ball']['drag'] = False

    if type(noise_fixed) == str:
        noise_fixed = [noise_fixed]

    if noise_fixed is None:
        noise_fixed = []

    if len(noise_fixed) > 0:
        noise_fixed = copy.copy(noise_fixed)

    if 'drag' in noise_fixed:
        p['Ball']['drag'] = True
        noise_fixed.remove('drag')

    if 'observation' in noise_fixed:
        p['Ball']['observation_noise_std'] = [0.01, 0.01, 0.01]
        noise_fixed.remove('observation')

    if 'delay' in noise_fixed:
        delay = NOISE_LEVELS["delay"]["med"]  # 30
        delay_dt = compute_delay_steps(delay, dictionaries['BallCatching']['framerate'])
        p['Agent']['delay'] = int(delay_dt)
        noise_fixed.remove('delay')
        print("DELAYING %d" % p['Agent']['delay'])

    if len(noise_fixed) != 0:
        raise Exception("noise source not supported in single experiment: %s" % str(noise_fixed))

    # worst case (w/o drag)
    # p['Ball']['observation_noise_std'] = [0.1,0.1,0.]
    # p['Ball']['system_noise_std'] = [0.1,0.1,0.]
    # p['Agent']['noise_std'] = [0.1,0.1,0.]
    # p['Ball']['wind_gust_force'] = NOISE_TYPES["wind"]['neg']
    # p['Ball']['drag'] = False


    # ax, V = 15, 40
    # ax, V = -15, 20
    p['Agent']['x_init'] = [float(ax), 0., float(0)]
    p['Agent']['x_init_type'] = 'cartesian'
    p['Ball']['v_init'] = [float(theta), 0., float(V)]

    if three_d:
        # aphi = np.deg2rad(10.)
        p = initialize_params_3d(p, ax, aphi)
    else:
        p['BallCatching']['dim'] = 2

    real_strategy_name, p = resolve_strategy(strategy, p)
    p['BallCatching']['strategy'] = real_strategy_name

    # print "Real strategy: %s" % real_strategy_name
    # Overriding parameters
    if len(parameters) > 0:
        print("Override:")
        print
        parameters
    update_dictionary(parameters, p, verbose=False)

    strategy = p['BallCatching']['strategy']
    experiment_name = "%s_default_%s" % (strategy, generate_timestamp())

    p['BallCatching']['trials'] = trials

    if real_strategy_name == 'COVStrategy':
        p['COVStrategy']['bearing_control'] = three_d
        # p['COVStrategy']['bearing_control'] = False

    experiment_sets = generate_experiment_sets(strategy, "3d" if three_d else "2d", [p], experiment_sets)
    # experiment_sets[experiment_name] = parametrizations

    experiment_container_root = ""
    return experiment_container_root, experiment_sets


# =======================================================================
# Full set including different sources of noise
# V and agent_x_init experiments -> 2D
def multi_experiment(strategies, ball_range, parameters={}, parameter_ranges=None,
                     trials=5, noise_fixed=None, container_name="", three_d=False,
                     noise_only=False):
    global dictionaries

    experiment_container_root = ""
    experiment_sets = {}

    if three_d:
        dictionaries['BallCatching']['dim'] = 3
        experiment_type = "3D"
        SETTINGS = EXPERIMENT_RANGES_3D
        varying_parameters = []

        if ball_range not in SETTINGS:
            raise (Exception("Ball range %s not supported" % ball_range))

        # set the Agent/Ball parameters you vary
        if len(SETTINGS[ball_range]['ax']) > 1:
            varying_parameters.append(("Agent/x_init/2", "$D$"))
        if len(SETTINGS[ball_range]['V']) > 1:
            varying_parameters.append(("Ball/v_init/2", "$V$"))
        if len(SETTINGS[ball_range]['aphi']) > 1:
            varying_parameters.append(("Agent/x_init/0", "$\phi$"))


    else:  # 2D
        dictionaries['BallCatching']['dim'] = 2
        experiment_type = "2D"
        SETTINGS = EXPERIMENT_RANGES_2D

        # set the two (!) Agent/Ball parameters you vary
        varying_parameters = [
            ("Agent/x_init/0", "$a_x$"),
            ("Ball/v_init/2", "$V$")
        ]

    experiment_container_root = experiment_type + "BallCatching" + "_" + container_name + "_" + generate_timestamp()

    print
    "Multi experiment 2D"
    print
    " ball_range: %s" % ball_range
    print
    " noise fixed: %s" % str(noise_fixed)

    if strategies is None or len(strategies) == 0:
        if three_d:
            raise Exception("Evaluation of all strategies not supported in 3D")
        print
        "evaluating ALL strategies (cpp only)!"
        strategies = STRATEGIES_CPP
        print(strategies)

    if parameter_ranges is not None and len(parameter_ranges) > 0:
        for k in parameter_ranges.keys():
            varying_parameters.append((k, k))

            # varying_parameters = [
            # ("COVStrategy/rv_delay", "rv_delay"),
            # ("COVStrategy/rv_averaging", "rv_averaging"),
            # ("Agent/x_init/0", "$a_x$"),
            # ("Ball/v_init/2", "$V$"),
            # ]

    ## standard params
    # drag = False
    # noise_std = [0.0]*3
    # wind_gust_force = [0.0]*3
    # agent_noise_std = [0.0]*3
    # noise_type = "system" # default
    ##noise_type = "observation"

    p_settings = {}

    p_settings["wind"] = [("off", NOISE_PARAMS_OFF["wind"])]
    p_settings["drag"] = [("off", NOISE_PARAMS_OFF["drag"])]
    p_settings["agent"] = [("off", NOISE_PARAMS_OFF["agent"])]
    p_settings["system"] = [("off", NOISE_PARAMS_OFF["system"])]
    p_settings["observation"] = [("off", NOISE_PARAMS_OFF["observation"])]
    p_settings["delay"] = [("off", NOISE_PARAMS_OFF["delay"])]

    if noise_fixed is not None:
        if type(noise_fixed) == str:
            noise_fixed = [noise_fixed]

        nfl = noise_fixed[0].lower()
        if "full_" in nfl or noise_fixed[0].lower() == "all":
            if noise_fixed[0].lower() == "all":
                p_settings["drag"] += NOISE_TYPES["drag"].items()
                p_settings["wind"] += NOISE_TYPES["wind"].items()
                p_settings["agent"] += NOISE_TYPES["agent"].items()
                p_settings["system"] += NOISE_TYPES["system"].items()
                p_settings["observation"] += NOISE_TYPES["observation"].items()
                p_settings["delay"] += NOISE_TYPES["delay"].items()
            elif nfl == "full_gaussian":
                p_settings["agent"] += NOISE_TYPES["agent"].items()
                p_settings["system"] += NOISE_TYPES["system"].items()
                p_settings["observation"] += NOISE_TYPES["observation"].items()
            elif nfl == "full_gaussian_drag":
                p_settings["agent"] += NOISE_TYPES["agent"].items()
                p_settings["system"] += NOISE_TYPES["system"].items()
                p_settings["observation"] += NOISE_TYPES["observation"].items()
                p_settings["drag"] += NOISE_TYPES["drag"].items()
            elif nfl == "full_gaussian_delay":
                p_settings["agent"] += NOISE_TYPES["agent"].items()
                p_settings["system"] += NOISE_TYPES["system"].items()
                p_settings["observation"] += NOISE_TYPES["observation"].items()
                p_settings["delay"] += NOISE_TYPES["delay"].items()
            elif nfl == "full_comb":
                p_settings["drag"] += NOISE_TYPES["drag"].items()
                p_settings["wind"] += NOISE_TYPES["wind"].items()
                p_settings["agent"] += NOISE_TYPES["agent"].items()
                p_settings["system"] += NOISE_TYPES["system"].items()
                p_settings["observation"] += NOISE_TYPES["observation"].items()
                p_settings["delay"] += NOISE_TYPES["delay"].items()

                # else:
                # print "Setting noise fixed to: %s" % noise_fixed
                # p_settings[noise_fixed] += NOISE_TYPES[noise_fixed].items()

        else:
            for nf in noise_fixed:
                print
                "Adding noise type to: %s --> %s" % (nf, str(NOISE_TYPES[nf].items()))
                if noise_only:
                    p_settings[nf] = []
                p_settings[nf] += NOISE_TYPES[nf].items()

    else:
        print
        "No noise"

    # --------
    # generate all experiment variants

    # all_noise_variant_created = False

    for wind_gust_force_type, wind_gust_force in p_settings["wind"]:
        # has_wind = wind_gust_force != NOISE_PARAMS_OFF["wind"]
        has_wind = wind_gust_force_type != "off"

        for delay_type, delay in p_settings["delay"]:
            # has_delay = delay != NOISE_PARAMS_OFF["delay"]
            has_delay = delay_type != "off"

            for drag_type, drag in p_settings["drag"]:
                # has_drag = drag != NOISE_PARAMS_OFF["drag"]
                has_drag = drag_type != "off"

                # if noise_fixed == ["all"] and has_drag and has_wind:
                ## hacky: in 'all' we only allow certain combinations
                # if all_noise_variant_created:
                # continue

                for observation_noise_std_type, observation_noise_std in p_settings['observation']:
                    # has_observation_noise = observation_noise_std != NOISE_PARAMS_OFF['observation']
                    has_observation_noise = observation_noise_std_type != "off"

                    for system_noise_std_type, system_noise_std in p_settings['system']:
                        # has_system_noise = system_noise_std != NOISE_PARAMS_OFF['system']
                        has_system_noise = system_noise_std_type != "off"

                        # if noise_fixed == ["all"] and has_observation_noise and has_system_noise:
                        ### hacky: in 'all' we only allow certain combinations
                        # if all_noise_variant_created:
                        # continue

                        for agent_noise_std_type, agent_noise_std in p_settings["agent"]:
                            # has_agent_noise = agent_noise_std != NOISE_PARAMS_OFF['agent']
                            has_agent_noise = agent_noise_std_type != "off"

                            # print "wind_gust_force_type, delay_type, drag_type, observation_noise_std_type, system_noise_std_type, agent_noise_std_type"
                            # print wind_gust_force_type, delay_type, drag_type, observation_noise_std_type, system_noise_std_type, agent_noise_std_type

                            # if noise_fixed == ["all"] and (has_observation_noise or has_system_noise) and has_agent_noise:
                            # if all_noise_variant_created:
                            # continue

                            # filter illegal variations and let "all noise sources active" happen once

                            # all_noise_variant_created_now = (has_agent_noise and has_system_noise and has_observation_noise and has_wind and has_drag)

                            # if len(NOISE_LEVELS.keys()) == 1 and NOISE_LEVELS.keys()[0] == "low":
                            #  # HACKY: delay is true and false if we have setting low - because in this setting these is no delay
                            #  has_delay = True

                            # we want to worst case both with and without drag
                            all_noise_variant_created_now = (
                            has_agent_noise and has_system_noise and has_observation_noise and has_wind and has_delay)

                            # for "full_comb" evaluate only the worst-case settings
                            if noise_fixed == ["full_comb"] and not all_noise_variant_created_now:
                                continue

                            # for "full_gaussian" evaluate only the worst-case gaussian
                            if noise_fixed == ["full_gaussian"] and not (
                                    has_agent_noise and has_system_noise and has_observation_noise):
                                continue

                            # for "full_gaussian_drag" evaluate only the worst-case gaussian
                            if noise_fixed == ["full_gaussian_drag"] and not (
                                        has_agent_noise and has_system_noise and has_observation_noise and has_drag):
                                continue

                            # for "full_gaussian_delay" evaluate only the worst-case gaussian
                            if noise_fixed == ["full_gaussian_delay"] and not (
                                        has_agent_noise and has_system_noise and has_observation_noise and has_delay):
                                continue

                            if all_noise_variant_created_now:
                                # create all variant only if levels agree
                                if (noise_fixed != ["all"] \
                                            and noise_fixed != ["full_comb"] \
                                            and noise_fixed != ["full_gaussian"]
                                    and noise_fixed != ["full_gaussian_drag"] \
                                            and noise_fixed != ["full_gaussian_delay"]) \
                                        or \
                                                agent_noise_std_type != observation_noise_std_type or \
                                                agent_noise_std_type != system_noise_std_type or \
                                                delay_type != system_noise_std_type:
                                    continue

                                all_noise_variant_created = all_noise_variant_created_now
                                print
                                "all_noise_variant_created"

                            elif (has_agent_noise and has_system_noise) or (
                                has_system_noise and has_observation_noise) or \
                                    (has_agent_noise and has_observation_noise) or \
                                    (has_wind and has_drag):
                                if noise_fixed != ["full_gaussian"] and noise_fixed != [
                                    "full_gaussian_drag"] and noise_fixed != ["full_gaussian_delay"]:
                                    continue
                                elif agent_noise_std != observation_noise_std or agent_noise_std != system_noise_std:
                                    continue

                            if noise_only and has_delay and "low" in NOISE_TYPES["delay"] \
                                    and NOISE_TYPES["delay"]["low"] == NOISE_PARAMS_OFF["delay"] \
                                    and delay_type == "low" and \
                                    not (has_agent_noise or has_system_noise or has_observation_noise or has_wind):
                                print("Delay low == delay off == ideal -- skipping!")
                                # raise Exception("")
                                continue

                            for strategy in strategies:
                                parametrizations = []

                                # parameter setting
                                if three_d:
                                    aphi_range = SETTINGS[ball_range]['aphi']
                                else:
                                    aphi_range = [None]

                                ax_range = SETTINGS[ball_range]['ax']
                                V_range = SETTINGS[ball_range]['V']

                                if 'drag' in SETTINGS[ball_range]:
                                    print("Overwriting drag setting by '%s'" % ball_range)
                                    if type(SETTINGS[ball_range]['drag']) == bool:
                                        print("Single value")
                                        drag = SETTINGS[ball_range]['drag']
                                    else:
                                        print("Multi value")
                                        drag = False
                                        if parameter_ranges is None:
                                            parameter_ranges = {}
                                        parameter_ranges['Ball/drag'] = SETTINGS[ball_range]['drag']
                                        varying_parameters.append(('Ball/drag', 'Ball/drag'))

                                trials_this = trials
                                # if np.linalg.norm(observation_noise_std) == 0. and np.linalg.norm(agent_noise_std) == 0.:
                                if not has_agent_noise and not has_system_noise and not has_observation_noise:
                                    print("  - No noise; 1 trial")
                                    trials_this = 1

                                for extra_params in param_range_dict_iterator(parameter_ranges):
                                    for ax in ax_range:
                                        for V in V_range:
                                            for aphi in aphi_range:
                                                delay_dt = compute_delay_steps(delay, dictionaries['BallCatching'][
                                                    'framerate'])

                                                # attention: yaml does not support numpy formats
                                                p = {}
                                                p['Agent'] = {'x_init': [float(ax), 0., 0.],
                                                              'noise_std': agent_noise_std,
                                                              'delay': int(delay_dt), }
                                                # 'delay': 0}
                                                p['Ball'] = {'v_init': [float(np.pi / 4), 0., float(V)],
                                                             'v_init_type': 'spherical',
                                                             'observation_noise_std': observation_noise_std,
                                                             'system_noise_std': system_noise_std,
                                                             'noise_dist_factor': 0.05,
                                                             'drag': drag,
                                                             'wind_gust_force': wind_gust_force
                                                             }
                                                if three_d:
                                                    p = initialize_params_3d(p, ax, aphi)

                                                else:
                                                    # IMPORTANT: 2D experiment; no z-noise allowed
                                                    p['Agent']['noise_std'][2] = 0.
                                                    p['Ball']['observation_noise_std'][2] = 0.
                                                    p['Ball']['system_noise_std'][2] = 0.

                                                # strategy renaming
                                                real_strategy_name, p = resolve_strategy(strategy, p)

                                                # FIXME hacky
                                                if three_d and real_strategy_name == 'COVStrategy':
                                                    p['COVStrategy']['bearing_control'] = three_d

                                                # finalize
                                                p['BallCatching'] = {'strategy': real_strategy_name,
                                                                     'dim': 3 if three_d else 2,
                                                                     'pause_on_catch': False,
                                                                     'varying_parameters': varying_parameters,
                                                                     'trials': trials_this}

                                                # update the dictionary with extra params (if any)
                                                if len(extra_params) > 0:
                                                    update_dictionary(extra_params, p)

                                                parametrizations.append(p)

                                es = generate_experiment_sets(strategy, experiment_type, parametrizations,
                                                              experiment_sets,
                                                              expose_params=[(k1, k2) for (k1, v) in
                                                                             extra_params.items() for k2 in v])
                                experiment_sets = dict(experiment_sets.items() + es.items())

    dictionaries['BallCatching']['visualize'] = False

    return experiment_container_root, experiment_sets