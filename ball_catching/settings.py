from collections import OrderedDict
import numpy as np

from ball_catching.utils.utils import create_dictionary

###################
# Atomic settings #
###################

# Different noise types and exact parameters

NOISE_LEVELS ={'low': [0.001]*3, 'med': [0.01]*3, 'high': [0.1]*3}

NOISE_TYPES = OrderedDict()
NOISE_TYPES["agent"] = NOISE_LEVELS
NOISE_TYPES["observation"] = NOISE_LEVELS
NOISE_TYPES["system"] = NOISE_LEVELS
NOISE_TYPES["drag"] = {"drag": True}
# wind negative and positive
NOISE_TYPES["wind"] = {"pos": [8.0, 2.0, 0.], "neg": [-8.0, 2.0, 0.]}

# 200 ms -> med
# 400 ms -> high
NOISE_TYPES["delay"] = {"low": 0.0, "med": 0.2, "high": 0.4} # in milliseconds!
#http://www.100fps.com/how_many_frames_can_humans_see.htm

# Parameter to use if certain noise type is switched off

NOISE_PARAMS_OFF = OrderedDict()
NOISE_PARAMS_OFF["agent"] = [0.0]*3
NOISE_PARAMS_OFF["observation"] = [0.0]*3
NOISE_PARAMS_OFF["system"] = [0.0]*3
NOISE_PARAMS_OFF["wind"] = [0.0]*3
NOISE_PARAMS_OFF["drag"] = False
NOISE_PARAMS_OFF["delay"] = 0


###############################################
#          Plotting Labels                    #
###############################################


STATS = ['mean(terminal_distance)', 'std(terminal_distance)',
         'mean(terminal_velocity)', 'std(terminal_velocity)',
         'mean(control_effort)', 'std(control_effort)',
         'mean(duration)', 'std(duration)',
         ]

########################################
# Multi experiment setting collections #
########################################

# If you select a "multi" experiment, these are pre-defined settings you can use as a "ragne"
#

EXPERIMENT_RANGES_2D = {
    # testing border cases only
    'simple': {'ax': tuple([5]), 'V': tuple([30.])},

    # testing border cases only
    'xxs': {'ax': tuple([-15., 15.]), 'V': tuple([20., 40.])},
    # testing border cases only
    'xs': {'ax': tuple([-15., 5, 15.]), 'V': tuple([20., 30, 40.])},
    # testing very coarse
    's': {'ax': np.arange(-15, 16., 7.5), 'V': np.arange(20, 41, 10.0)},
    # testing somewhat coarser
    'm': {'ax': np.arange(-15, 16., 5.0), 'V': np.arange(20, 41, 4.0)},
    # full kistemaker range: ax -> np.arange(-15, 16., 1.0)    V -> np.arange(20, 41, 1.0)
    'l': {'ax': np.arange(-15, 16., 1.0), 'V': np.arange(20, 41, 1.0)},

    # varying V negatively, too (throw in opposite direction)
    'xs-': {'ax': tuple([-15., 15.]), 'V': tuple([-40, -20, 20., 40.])},
    's-': {'ax': np.arange(-15, 16., 7.5),
           'V': np.concatenate([-np.arange(20, 41, 10.0), np.arange(20, 41, 10.0)])},
    'm-': {'ax': np.arange(-15, 16., 5.0),
           'V': np.concatenate([-np.arange(20, 41, 4.0), np.arange(20, 41, 4.0)])},

}

# --------
EXPERIMENT_RANGES_3D = {
    'simple': {'ax': tuple([5]), 'V': tuple([20.]), 'aphi': tuple([np.pi / 16])},

    'xs': {'ax': tuple([-15., 15.]), 'V': tuple([20., 40.]), 'aphi': tuple([np.pi / 16, np.pi / 2])},

    's': {'ax': np.arange(-15, 16., 7.5), 'V': np.arange(20, 41, 10.0),
          'aphi': tuple([np.pi / 16, np.pi / 8, np.pi / 4, np.pi / 2])},

    'm': {'ax': np.arange(-15, 16., 5.0), 'V': np.arange(20, 41, 4.0),
          'aphi': tuple([np.pi / 16, np.pi / 8, np.pi / 4, np.pi / 2])},
}


############################################
# Experiment and strategy default settings #
############################################

create_dictionary(
    'BallCatching', {
        'strategy': "",
        'trials': 2,
        'framerate': 60,
        'visualize': False,  # True
        'auto_camera': True,
        'pause_on_catch': False,
        'visualization_speed': 0.5,
        'debug_external': False,

        # for python backend: dimensionality of the scenario
        # 'dim': 2,
        # 'dim': 3,
    })

create_dictionary(
    'Agent', {
        'mass': 0.0,
        # here, delay must be provided in steps (i.e. framerate included!)
        # when using single_experiment or multi_experiment, you use milliseconds and it  is automatically converted
        # based on the system's framerate.
        'delay': 0,
        # initial position
        'x_init_type': 'cartesian',  # default: give a 3D vector representing x,y,z
        # spherical: give distance from (0,0,0) / ball landing point
        # and angle on plane
        # 'x_init': [15., 0.01, 16.],
        'x_init': [15.0, 0., 0.],
        # 'x_init': [13., 0., 0.],
        'x_relative': True,
        'x_relative_perfect': False,  # relative position to perfect trajectory
        # initial orientation quaterion x,y,z,w (optional, set to 0 for ignoring)
        'q_init': [0., 0., 0., 0.],
        # maximal velocity
        'v_max': 9.,  # 100000000000.,#9.,
        # maximal acceleration; if we assume a human can accelerate to top speed
        'a_max': 4.5,  # 100000000000., #4.5,

        'momentum_max': np.pi,  # let's assume we can turn 180 deg in 1 second
        # 'momentum_max': 2*np.pi,# let's assume we can turn 360 deg in 1 second

        'torque_max': np.pi,  # currently ignored

        'show_camera': False,
        'camera_offset': [0., 0., 0.],

        'noise_mean': [0., 0., 0.],
        # 'noise_std': [0.5, 0, 0],
        'noise_std': [0., 0., 0.],

    })

create_dictionary(
    'Ball', {
        'noise_mean': [0.0] * 3,
        # radius; football: 0.11m, baseball: 0.037m
        'radius': 0.0366,  # 0.037,
        # mass; football: 0.4kg, baseball: 0.15kg
        'mass': 0.15,  # 0.145
        'x_init': [0., 0., 0.],
        # 'v_init_type': 'cartesian', # default: give a 3D vector representing x,y,z
        # 'v_init': [15., 12., 15.],
        'v_init_type': 'spherical',  # give angle theta (elevation), phi (azimuth) and a length V
        'v_init': [float(np.pi / 4), 0., 30.],
        'drag': False,  # True,
        # geometric drag coefficient; football is 0.25 /baseball 0.5;
        # see http://www.grc.nasa.gov/WWW/k-12/airplane/balldrag.html
        # see http://www.grc.nasa.gov/www/k-12/airplane/socdrag.html
        # see http://wps.aw.com/wps/media/objects/877/898586/topics/topic01.pdf
        'drag_cw': 0.5,  # 0.3,
        'drag_rho': 1.293,  # 1.2, # air
        # noise std (system/observation)
        'observation_noise_std': [0.0] * 3,
        'system_noise_std': [0.0] * 3,
        # modulate observation stdev with distance: stdev_t := stdev * noiseDistFactor * distance
        'noise_dist_factor': 0.05,
        # wind gust disturbance
        'wind_gust_force': [0., 0., 0.],
        # 'wind_gust_force': [-8.0, 2.0, 0],
        'wind_gust_duration': 0.1,
        'wind_gust_relative_start_time': 0.4,

        'noise_dim': 2,  # FIXME: for python backend to simulate only z noise, for optimizing OAC3D

        'recorded_trajectory': None,
    })

create_dictionary(
    'LQRStrategy', {
        # deprecated parameters
        'drag_tti': False,
        'gain_folder': 'lqr',
        # 'gain_folder': "lqrV10000",
        'time_approx': False,
        'open_loop': False,
        # 'kf_2d': True,  # dangerous! inversion of matrices will be unstable
        'kf_2d': False,

        # default kalman filter parameters
        'kf_P0_ball': 10.,
        'kf_P0_agent': 0.1,
        'kf_P0_velocity_factor': 60.,
        'kf_Q_ball': 1e-5,
        'kf_Q_agent': 1e-5,
        'kf_Q_velocity_factor': 60.,
        'kf_R_ball': 1e-5,
        'kf_R_agent': 1e-5,
        'kf_R_velocity_factor': 60.,

        'use_kalman_filter': True,

        # costs: used by python backend to solve LQR
        'terminal_distance': 1000.,
        'terminal_velocity': 0.,
        'control_effort': 0.01,

    })

create_dictionary(
    'iLQRStrategy', {
        'drag_setting': 'baseball',

        # rest of the parameters is used from LQRStrategy
    })



create_dictionary(
    'COVOAC2DStrategy', {
        'framerate': 60,
        'rv_min_index': 3,
        'rv_delay': 50,
        'rv_averaging': 30,
        'dtheta_averaging': 5,
    })

create_dictionary(
    'COVOAC3DStrategy', {
        'rv_min_index': 3,
        'framerate': 60,
        'rv_delay': 50,  # 25, #0,
        'rv_averaging': 30,  # 10, # 6, # 1 means: no average
        'dtheta_averaging': 5,
        'cba_averaging': 10,  # 1,
        'cba_delay': 9,
        'beta_thresh': 0.017,
    })

create_dictionary(
    'AP2DStrategy', {
        'framerate': 60,
        'rv_min_index': 3,
        'theta_averaging': 0.,
        'ap_t': "explicit",
        'rv_delay': 50,  # 25, #0,
        'rv_averaging': 30,  # 10, # 6, # 1 means: no average
        'dtheta_averaging': 5,
    })

# ============================================================================
#                           MPC Strategy
# ============================================================================
create_dictionary(
    'MPCStrategy', {
        'active': True,
        'F_c1': 12.,
        'F_c2': 1e-15,
        'delay': 0,
        'M': 1e-15,
        'N': 1e-15,
        # 'w_max': 2*np.pi ,
    })