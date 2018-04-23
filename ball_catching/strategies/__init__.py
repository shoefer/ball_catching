from ball_catching.strategies.angular import AP2DStrategy, COVOAC2DStrategy, COVOAC3DStrategy
from ball_catching.strategies.cartesian_lqr import LQRStrategy, iLQRStrategy
from ball_catching.strategies.misc import ZeroStrategy

# from ball_catching.strategies.cartesian_mpc import MPCStrategy
try:
    from ball_catching.strategies.cartesian_mpc import MPCStrategy
except:
    import warnings
    warnings.warn("MPC not supported. Follow special installation instructions in README")
    MPCStrategy = None


STRATEGIES = {
    'ZeroStrategy': (ZeroStrategy, {}),

    'AP2DStrategy': (
    AP2DStrategy, {'theta_averaging': 0., 'ddtheta_min': 0., 'rv_delay': 1, 'rv_averaging': 1, 'dtheta_averaging': 0}),
    # 'APIO2DStrategy': (AP2DStrategy, {'theta_averaging': 0., 'ddtheta_min': 0., 'rv_delay': 1e5, }),
    'APIO2DStrategy': (AP2DStrategy, {'theta_averaging': 0.,
                                      'dtheta_averaging': 0.,
                                      'rv_delay': 1e5, }),
    'APOAC2DStrategy': (AP2DStrategy, {'theta_averaging': 0.,
                                       'dtheta_averaging': 0.,
                                       }),
    'APConst2DStrategy': (AP2DStrategy, {'ddtheta_min': 0., 'rv_delay': -1, 'rv_const': 0.2}),
    'APIdeal2DStrategy': (AP2DStrategy, {'ddtheta_min': 0., 'theta_averaging': 0., 'rv_delay': -1, 'use_rv_opt': True}),
    'APIdealImplicitT2DStrategy': (
    AP2DStrategy, {'ddtheta_min': 0., 'theta_averaging': 0., 'rv_delay': -1, 'use_rv_opt': True, 'ap_t': "implicit"}),

    'OAC2DStrategy': (COVOAC2DStrategy, {'ddtheta_min': 0., 'rv_delay': 1, 'rv_averaging': 1, 'dtheta_averaging': 0}),
    'OAC3DStrategy': (COVOAC3DStrategy, {'ddtheta_min': 0., 'rv_delay': 1, 'rv_averaging': 1, 'dtheta_averaging': 0}),
    'COVIO2DStrategy': (COVOAC2DStrategy, {'rv_delay': 1e5}),
    'COVIO3DStrategy': (COVOAC3DStrategy, {'rv_delay': 1e5}),
    'COVConst2DStrategy': (COVOAC2DStrategy, {'rv_delay': -1, 'rv_const': 0.208498}),
    'COVConst3DStrategy': (COVOAC3DStrategy, {'rv_delay': -1, 'rv_const': 0.2}),
    'COVOAC2DStrategy': (COVOAC2DStrategy, {}),
    'COVOAC3DStrategy': (COVOAC3DStrategy, {}),
    'COVIdeal2DStrategy': (COVOAC2DStrategy, {'rv_delay': -1, 'use_rv_opt': True}),

    'LQRStrategy': (LQRStrategy, {'use_kalman_filter': False}),
    'LQGStrategy': (LQRStrategy, {'use_kalman_filter': True}),

    # Uncomment if you want to try out LQR/LQG with different terminal velocity costs
    # 'LQGV10Strategy':
    #     (LQRStrategy, {'use_kalman_filter': True, 'terminal_velocity': 10}),
    # 'LQGV1000Strategy':
    #     (LQRStrategy, {'use_kalman_filter': True, 'terminal_velocity': 1000}),

    'iLQRStrategy': (iLQRStrategy, {'use_kalman_filter': False}),
    'iLQGStrategy': (iLQRStrategy, {'use_kalman_filter': True}),

    'iLQRSoccerStrategy':
        (iLQRStrategy, {'use_kalman_filter': False, 'drag_setting': 'soccerball'}),
    'iLQGSoccerStrategy':
        (iLQRStrategy, {'use_kalman_filter': True, 'drag_setting': 'soccerball'}),

    # ----------------------------------------------------

    'MPCStrategy':
        (MPCStrategy, {'active': True, 'F_c1': 7.5, 'F_c2': 7.5, 'M': 1e-15, 'delay': 0, }),
    'MPCStrategyHighUncertainty':
        (MPCStrategy, {'active': True, 'F_c1': 7.5, 'F_c2': 7.5, 'M': 1e-3, 'delay': 0, }),

    # ----------------------------------------------------
    # MPC Hyperparam search
    # 'MPCStrategyFDefM0':
    #     (MPCStrategy, {'active': True, 'F_c1': 7.5, 'F_c2': 2.5, 'M': 1e-15, 'delay': 0}),
    # 'MPCStrategyFDefM0Delayed':
    #     (MPCStrategy, {'active': True, 'F_c1': 7.5, 'F_c2': 2.5, 'M': 1e-15, 'delay': 1}),
    #
    # 'MPCStrategyFDefM0001':
    #     (MPCStrategy, {'active': True, 'F_c1': 7.5, 'F_c2': 2.5, 'M': 1e-3, 'delay': 0}),
    #
    # 'MPCStrategyF55M0':
    #     (MPCStrategy, {'active': True, 'F_c1': 5., 'F_c2': 5., 'M': 1e-15, 'delay': 0, }),
    # 'MPCStrategyF55M0001':
    #     (MPCStrategy, {'active': True, 'F_c1': 5., 'F_c2': 5., 'M': 1e-3, 'delay': 0, }),
    #
    # 'MPCStrategyF7575M0':
    #     (MPCStrategy, {'active': True, 'F_c1': 7.5, 'F_c2': 7.5, 'M': 1e-15, 'delay': 0, }),
    # 'MPCStrategyF12M0':
    #     (MPCStrategy, {'active': True, 'F_c1': 12., 'F_c2': 1e-15, 'M': 1e-15, 'delay': 0, }),
    # 'MPCStrategyF1212M0':
    #     (MPCStrategy, {'active': True, 'F_c1': 12., 'F_c2': 12., 'M': 1e-15, 'delay': 0, }),
    #
    # 'MPCStrategyF7575M0Nmax1':
    #     (MPCStrategy, {'active': True, 'F_c1': 7.5, 'F_c2': 7.5, 'M': 1e-15, 'N_max': 1., 'delay': 0, }),
    # 'MPCStrategyF12M0Nmax1':
    #     (MPCStrategy, {'active': True, 'F_c1': 12., 'F_c2': 1e-15, 'M': 1e-15, 'N_max': 1., 'delay': 0, }),
    # 'MPCStrategyF1212M0Nmax1':
    #     (MPCStrategy, {'active': True, 'F_c1': 12., 'F_c2': 12., 'M': 1e-15, 'N_max': 1., 'delay': 0, }),

}