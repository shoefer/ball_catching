# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:24:25 2016

@author: shoefer
"""

import numpy as np
import os
import pandas as pd

from ball_catching.utils.cont2discrete import cont2discrete

# -----
# state dim
#   0 -> xb
#   1 -> xb'
#   2 -> xb''
#   3 -> yb
#   4 -> yb'
#   5 -> yb''
#   6 -> zb
#   7 -> zb'
#   8 -> zb''
#
#   9 -> xa
#   10 -> xa'
#   11 -> xa''
#   12 -> za
#   13 -> za'
#   14 -> za''
STATE_DIM = 15

# -----
# action dim
ACTION_DIM = 2

# -----
# system

A = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 -> xb
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 -> xb'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 -> xb''
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3 -> yb
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 -> yb'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 -> yb''
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 6 -> zb
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 7 -> zb'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8 -> zb''
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 9 -> xa
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10 -> xa'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11 -> xa''
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 12 -> za
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13 -> za'
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14 -> za''
              ])

# A = np.array([[ 0,	1,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 0 -> xb
#			        [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 1 -> xb'
#			        [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 2 -> xb''
#      			  [ 0,	0,	0, 0,	1,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 3 -> yb
#      			  [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],  # 4 -> yb'
#			        [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 5 -> yb''
#      			  [ 0,	0,	0, 0,	0,	0, 0,	1,	0, 0,	0,	0,	0, 0, 0	],	# 6 -> zb
#      			  [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 7 -> zb'
#			        [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 8 -> zb''
#      			  [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	1,	0,	0, 0, 0	],	# 9 -> xa
#      			  [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 10 -> xa'
#      			  [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0	],	# 11 -> xa''
#      			  [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 1, 0	],	# 12 -> za
#      			  [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0 ],	# 13 -> za'
#      			  [ 0,	0,	0, 0,	0,	0, 0,	0,	0, 0,	0,	0,	0, 0, 0 ],	# 14 -> za''
#      			  ])

# acceleration-based control
Bacc = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, ],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ]]).T

# velocity-based control
Bvel = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, ],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ]]).T

# observability is useful for Kalman filtering -> C only observes positions & agent velocities
C = np.array([
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # xb
  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # yb
  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # zb
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # xa
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # xa'
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # za
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # za'
])

D = 0


# ----------------------------------------------------------------------------------------

class DynamicsModelType(type):
  def __call__(cls, *args, **kwargs):
    try:
      if len(args) == 0 and len(kwargs) == 0:
        return cls.__instance
      else:
        raise AttributeError()
    except AttributeError:
      dm = super(DynamicsModelType, cls).__call__(*args, **kwargs)
      if "copy" in kwargs and kwargs["copy"]:
        print ("WARN: Generating copy of DynamicsModel, not resetting global one")
        return dm

      try:
        if cls.__instance is not None:
          print ("INFO: Resetting global dynamics model")
      except:
        pass

      cls.__instance = dm
      return cls.__instance


class DynamicsModel:
  __metaclass__ = DynamicsModelType

  def __init__(self, dt=None, framerate=None, gravity=9.81,
               v_max=9., a_max=4.5,
               rho=1.293, c=0.5, r=0.0366, mass=0.15,
               sigma_ball=0., sigma_agent=0.,
               drag=False,
               dim=None,
               wind_gust_force=[0., 0., 0.],
               wind_gust_duration=0.,
               wind_gust_relative_start_time=0.,
               copy=False):
    """
    Singleton dynamics model.

    If you want to create a new dynamics model w/o overwriting
    the current global instance, then set "copy=True".
    """

    self.DT = dt
    self.FRAMERATE = framerate
    self.GRAVITY = gravity

    # agent properties
    self.AGENT_V_MAX = v_max
    self.AGENT_A_MAX = a_max

    # drag-relevant ball dynamics
    self.rho = rho
    self.c = c
    self.r = r
    self.mass = mass
    # necessary for drag
    self.A = np.pi * self.r * self.r
    self.sigma_ball = sigma_ball
    self.sigma_agent = sigma_agent

    self.wind_gust_force = wind_gust_force
    self.wind_gust_duration = wind_gust_duration
    self.wind_gust_relative_start_time = wind_gust_relative_start_time

    # dimensionality: 2d or 3d (for sampling noise)
    self.dim = dim
    assert (self.dim is not None)
    self.drag = drag

    # discretized system matrices
    self.Adt = None
    self.Baccdt = None
    self.Bacc = None
    self.B = None
    self.Bdt = None
    self.cdt = None

    self.Bvel = None
    self.Cdt = None
    self.Ddt = None

    self.compute_J_drag = None

    self.copy = copy
    self._generate_dynamics()

  def _generate_dynamics(self):
    if self.DT is not None:
      assert (self.FRAMERATE is None)
      self.FRAMERATE = 1. / self.DT
    else:
      assert (self.FRAMERATE is not None)
      self.DT = 1. / self.FRAMERATE

    # discretize system
    self.Adt, self.Baccdt, self.Cdt, self.Ddt, dt = cont2discrete((A, Bacc, C, D), self.DT)
    self.Adt, self.Bveldt, self.Cdt, self.Ddt, dt = cont2discrete((A, Bvel, C, D), self.DT)

    # default: acceleration driven dynamics
    self.B = Bacc
    self.Bdt = self.Baccdt

    # constant offset in ddy: gravity
    self._GRAVITY_VECTOR = np.array([0, -self.GRAVITY, 0])
    self.cdt = np.zeros(self.Adt.shape[0])
    self.cdt[5] = -self.GRAVITY
    # hacky: set ddx(t) = 0 (because it is covered by constant offset)
    self.Adt[2, 2] = self.Adt[5, 5] = self.Adt[8, 8] = 0

    if self.copy:
      print ("---------")
      print ("Local Dynamics: ")
    else:
      print ("=========")
      print ("GLOBAL Dynamics: ")

    print ("Dimensionality: %d " % self.dim)
    print ("  DT=%.5f, FRAMERATE=%.1f" % (self.DT, self.FRAMERATE))
    print ("  drag=%s" % (self.drag))
    print ("Agent: ")
    print ("  AGENT_A_MAX = %.2f" % self.AGENT_A_MAX)
    print ("  AGENT_V_MAX = %.2f" % self.AGENT_V_MAX)
    print ("Ball: ")
    print ("  radius = %.5f" % self.r)
    print ("  mass   = %.5f" % self.mass)
    print ("  c      = %.5f" % self.c)
    if self.copy:
      print ("---------")
    else:
      print ("=========")

  def is_equivalent(self, dyn):
    return np.all([
      self.DT == dyn.DT,
      self.FRAMERATE == dyn.FRAMERATE,
      self.GRAVITY == dyn.GRAVITY,
      self.AGENT_V_MAX == dyn.AGENT_V_MAX,
      self.AGENT_A_MAX == dyn.AGENT_A_MAX,
      self.rho == dyn.rho,
      self.c == dyn.c,
      self.r == dyn.r,
      self.mass == dyn.mass,
      self.drag == dyn.drag,
      self.dim == dyn.dim,
    ])

  def set_presimulate(self):
    self._presimulate = True
    self._epsilons = []

  def set_run(self):
    self._presimulate = False
    # revert episilon for "popping"
    # print ("len(self._epsilons)", len(self._epsilons))
    self._epsilons = list(reversed(self._epsilons))

  def set_stop(self):
    if not self.is_nonlinear():
      return True

    sn = len(self._epsilons) == 0
    self._epsilons = []
    return sn

  def _sample_system_noise(self):
    if self.is_nonlinear() and not self._presimulate:
      return self._epsilons.pop()

    s_ball = np.random.randn(3) * self.sigma_ball
    s_agent = np.random.randn(2) * self.sigma_agent

    # we assume we are applying a random FORCE to the ball
    # but we add acceleration to the model, so we need to convert:
    # a = F/m
    s_ball /= self.mass

    s = np.zeros(STATE_DIM)
    # affects position
    # dims_ball = [0, 3, 6]
    # dims_agent = [9, 12]

    # affects acceleration
    dims_ball = [2, 5, 8]
    dims_agent = [11, 14]

    # assign
    s[dims_ball] = s_ball
    s[dims_agent] = s_agent
    # check dimensionality
    if self.dim == 2:
      s[dims_ball[-1]] = 0.
      s[dims_agent[-1]] = 0.

    if self.has_wind():
      s += self.noise_wind_current
      self.noise_wind_current[:] = 0.  # delete

    if self.is_nonlinear():
      self._epsilons.append(s)

    return s

  def is_nonlinear(self):
    return self.sigma_ball > 0 or self.drag or self.has_wind

  def is_ball_on_ground(self, x_t):
    return x_t[3] < 0

  def step(self, x_t, u_t, noise=True):
    if self.drag:
      return self.step_drag(x_t, u_t, noise=noise)
    else:
      return self.step_linear(x_t, u_t, noise=noise)

  def step_linear(self, x_t, u_t, noise=True):
    """ Evaluate the system x and u at one time step """
    # linear system
    # sys.stdout.write("["+str((sgm)) + "] \n")
    # sys.stdout.flush()

    cdt = self.cdt
    if len(x_t.shape) == 2:
      cdt = cdt.reshape((-1, 1))
    else:
      cdt = cdt.reshape((-1,))

    x = np.dot(self.Adt, x_t) + np.dot(self.Bdt, u_t) + cdt

    if noise:
      # ball: set acceleration due to gravity (because might get overwritten)
      # due to our constant acceleration model
      # x[2], x[5], x[8] = self._GRAVITY_VECTOR # FIXME

      sgm = self._sample_system_noise()
      if np.any(sgm != 0.):
        # add noise
        x += sgm

        # sys.stdout.write(""+str(x) + " \n")
        # sys.stdout.write("" + str(sgm) + " \n")
        # sys.stdout.flush()

    return x

  def step_drag(self, x_t, u_t, noise=True):
    """
    Compared to
      http://www.livephysics.com/simulations/mechanics-sim/projectile-motion-simulation/
    """
    # x = self.step_linear(x_t, u_t)
    # x = np.dot(self.Adt, x_t) + np.dot(self.Bdt, u_t)

    x_t_ = np.asarray(x_t).reshape((-1,))
    u_t_ = np.asarray(u_t).reshape((-1,))

    x = np.dot(self.Adt, x_t_) + np.dot(self.Bdt, u_t_) + self.cdt.reshape((-1,))

    v = np.array([x[1], x[4], x[7]])
    # new acceleration
    # x[2], x[5], x[8] = self._GRAVITY_VECTOR - v*v * 0.5 * self.rho * self.c * self.A/self.mass
    ddx = - v * v * 0.5 * self.rho * self.c * self.A / self.mass
    for i, idx in enumerate([2, 5, 8]):
      x[idx] += ddx[i]

    if noise:
      sgm = self._sample_system_noise()
      # sys.stdout.write("["+str(max(sgm)) + "] ")
      # sys.stdout.flush()
      x += sgm

    return x

  def has_wind(self):
    return np.any(np.abs(self.wind_gust_force)) > 0 and self.wind_gust_duration > 0.

  def precompute_trajectory(self, x0):
    # get linear duration
    t_n, N, x_n, z_n = self.get_time_to_impact(x0)
    fr = DynamicsModel().FRAMERATE

    if not self.is_nonlinear():
      # nothing to do
      return t_n, N, x_n, z_n

    self.set_presimulate()

    if self.has_wind():
      assert (self.wind_gust_relative_start_time >= 0. and self.wind_gust_relative_start_time < 1.)
      self.wind_gust_start_time = t_n * self.wind_gust_relative_start_time
      # self.wind_gust_end_time = t_n * (self.wind_gust_relative_start_time+self.wind_gust_duration)
      self.wind_gust_end_time = t_n * (self.wind_gust_relative_start_time) + self.wind_gust_duration
      print ("Dynamics: wind_gust -> %.2f s to %.2f s   -> %d to %d" \
             % (self.wind_gust_start_time, self.wind_gust_end_time, round(self.wind_gust_start_time / fr),
                round(self.wind_gust_end_time / fr)))
      self.noise_wind_current = np.zeros(STATE_DIM)

    dims_ball = [2, 5, 8]  # FIXME copied

    # we need to pre-simulate
    x_ = x0.reshape((-1,))
    i = 0
    while i == 0 or x_[3] > 0.:
      tcur = i / fr

      # wind
      if self.has_wind():
        self.noise_wind_current[:] = 0.
        if tcur > self.wind_gust_start_time and tcur < self.wind_gust_end_time:
          self.noise_wind_current[dims_ball] = self.wind_gust_force
          self.noise_wind_current[dims_ball] /= self.mass  # it's a force
          # print ("WIND! %d, %f" % (i, tcur, ))
          # print self.noise_wind_current[dims_ball]

      x_ = self.step(x_, [0., 0.]).reshape((-1,))
      i += 1

    # print "last x ", x_[3]

    t_n = i / fr
    x_n = x_[0]
    z_n = x_[6]

    self.set_run()

    return t_n, i, x_n, z_n

  def get_time_to_impact(self, x0, ignore_drag=False):
    """
        Returns time to impact related variables as tuple:
          - t seconds
          - N steps at current framerate
          - x position of ball
          - z position of ball

    """

    drag = self.drag
    if ignore_drag:
      drag = False

    x0 = x0.flatten()

    if x0[3] < 0:
      return 0, 0, 0, 0

    if not drag:
      g = self.GRAVITY
      a, b, c = -g / 2, x0[4], x0[3]

      phalf = - b / (2.0 * a)
      pm_term = np.sqrt((b ** 2) / (4 * a ** 2) - c / a)
      t_n = phalf + pm_term
      x_n = x0[1] * t_n + x0[0]
      z_n = 0.

    else:
      # dynamics.set_presimulate()

      # we need to pre-simulate
      x_ = x0.reshape((-1,))
      i = 0
      while i == 0 or x_[3] > 0.:
        x_ = self.step_drag(x_, [0., 0.], noise=False).reshape((-1,))
        i += 1

      t_n = i / self.FRAMERATE
      x_n = x_[0]
      z_n = x_[6]

    assert (not np.isnan(t_n))

    # t_n seconds at current FRAMERATE
    N = int(np.ceil(t_n * self.FRAMERATE))

    return t_n, N, x_n, z_n

  def compute_J(self, x_t, u_t):
    if self.drag:
      if self.compute_J_drag is None:
        self._derive_drag_jacobian()
      return np.asarray(self.compute_J_drag(x_t, u_t))
    else:
      return self.Adt

  def _derive_drag_jacobian(self):
    # dt = self.DT
    g = self.GRAVITY
    rho, c, Ac, mass = self.rho, self.c, self.A, self.mass

    import sympy as sp
    from sympy.abc import x, y, z
    # from sympy import symbols, Matrix

    dx, dy, dz, ddx, ddy, ddz = sp.symbols("dx, dy, dz, ddx, ddy, ddz")
    ax, az, dax, daz, ddax, ddaz = sp.symbols("ax, az, dax, daz, ddax, ddaz")
    X = sp.Matrix([x, dx, ddx, y, dy, ddy, z, dz, ddz,
                   ax, dax, ddax, az, daz, ddaz, ])

    ux, uz = sp.symbols("ux, uz")
    U = sp.Matrix([ux, uz])

    A = sp.Matrix(self.Adt)
    B = sp.Matrix(self.Bdt)
    f_xu = sp.Matrix(A.dot(X)) + sp.Matrix(B.dot(U))
    # drag
    v = sp.Matrix([dx, dy, dz])

    # wrong: dd* does not evolve depending on the previous time step
    # f_xu[2], f_xu[5], f_xu[8] = sp.Matrix([ddx, ddy, ddz]) \

    # Correct but -g is constant and will thus disappear from Jacobian
    # -> put it into starting state
    # f_xu[2], f_xu[5], f_xu[8] = sp.Matrix([0, -g, 0]) \
    #  - v.multiply_elementwise(v) * 0.5 * rho * c * Ac/mass

    # Correct: ddx and ddz solely depend on squared velocity
    f_xu[2], f_xu[5], f_xu[8] = \
      - v.multiply_elementwise(v) * 0.5 * rho * c * Ac / mass

    self.FJ_drag = f_xu.jacobian(sp.Matrix([X]))
    self._compute_J_drag = sp.lambdify((dx, dy, dz), self.FJ_drag)
    self.compute_J_drag = lambda x, _: self._compute_J_drag(x[1], x[4], x[7])

  def observe_state(self, x_t):
    return self.C.dot(x_t)

  @property
  def state_dim(self):
    return self.Adt.shape[0]

  @property
  def action_dim(self):
    return self.Bdt.shape[1]

  @property
  def observation_dim(self):
    return self.Cdt.shape[0]


# ----------------------------------------------------------------------------------------

class RecordedTrajectoryStepper:
  #required_cols = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", ]
  required_cols = ["x", "vx", "ax", "y", "vy", "ay", "z", "vz", "az", ]

  def __init__(self, pd_pkl, dt=None):
    if dt is None:
      DT = dt = DynamicsModel().DT
    
    self.tj = pd.read_pickle(pd_pkl)
    assert (np.all([col in self.tj.columns for col in self.required_cols ]))
    
    if "t" in self.tj.columns:
      # downsample if necessary
      dt = round(self.tj.iloc[1].t - self.tj.iloc[0].t, 5)
      assert (round(DT, 5) == dt)
    
    # HACK: we need to append an additional column where the ball's height < 0
    if self.tj.iloc[-1].y > 0:
      xT = np.zeros((STATE_DIM,))
      xT = self._copy(self.tj.iloc[-1], xT)
      xTp1 = DynamicsModel().step(xT, np.zeros((ACTION_DIM,)))
      
      # copy back
      t_next = self.tj.iloc[-1].t+DT
      # create
      new_idx = len(self.tj)
      self.tj.loc[new_idx] = np.nan
      # set time
      self.tj.loc[new_idx,"t"] = t_next
      # set rest
      for c, v in zip(self.required_cols, xTp1[:9]):
        #self.tj.loc[t_next][c] = v
        self.tj.loc[new_idx,c] = v
      if self.tj.loc[new_idx,"y"] > 0:
        self.tj.loc[new_idx,"y"]*= -1

      path, basename = os.path.split(pd_pkl)
      fn_rsmpl = os.path.join(path, os.path.splitext(basename)[0] + ("_dt%.4f_fixed" % DT) + ".pkl")
      self.tj.to_pickle(fn_rsmpl)
    
    self.idx = 0
    self.t = 0

  def initialize_for_trial(self):
    pass

  def _copy(self, tj_t, x):
    x[:3] = tj_t.x, tj_t.vx, tj_t.ax
    x[3:6] = tj_t.y, tj_t.vy, tj_t.ay
    x[6:9] = tj_t.z, tj_t.vz, tj_t.az

    # v and a are NaN in the beginning
    x = np.nan_to_num(x)

    return x    
    
  def __call__(self, x_t, u_t):
    x = DynamicsModel().step(x_t, u_t)
    
    tj_t = self.tj.iloc[self.idx]
    x = self._copy(tj_t, x)

    assert (not np.any (np.isnan(x)))
    
    self.idx += 1
    
    return x

#-----------------------------------------------------------------------------------------


def parabola_find_null_points(a,b,c):
  # FIXME delete
  phalf = - b/(2.0*a)
  pm_term = np.sqrt( (b**2) / (4*a**2) - c/a)
  if b > 0:
    return (phalf+pm_term, phalf-pm_term)
  else:
    return (np.abs(phalf-pm_term), phalf+pm_term)

# ----------------------------------------------------------------------------------------

def observe_angle_and_distance(x_t):  
  # ball height
  yb = x_t[3]
  
  if yb < 0:
    return 0.,0.
  
  d = np.sqrt ( (x_t[0] - x_t[9])**2 + (x_t[3])**2 + (x_t[6] - x_t[12])**2)
  
  if d <= 0.:
    return 0.,0.
  
  # compute angle alpha
  alpha = np.arcsin( yb/d )
  if x_t[0]>x_t[9]:
    #d=-d
    alpha=np.pi-alpha
    pass
  
  return alpha,d
  
def normalize_angle(angle):
  if np.abs(angle) > np.pi:
    if angle > 0:
      angle = - (2*np.pi - angle)
    else:
      angle = 2*np.pi + angle
  return angle 


def compute_bearing_angle(xt):
  ba = np.array([xt[9]-xt[0], xt[12]-xt[6]])
  ba /= np.linalg.norm(ba)
  
  a2bt = np.arctan2(ba[0], ba[1])
  a2bt = normalize_angle(a2bt)
  
  return a2bt 
  
# ----------------------------------------------------------------------------------------

def observation_noise_ball(x, std=[0.01,0.01,0.01], noiseDistFactor=0.05, noise_dim=-1):
  std = np.array(std)
  if noiseDistFactor != 0.:
    # agent ball distance
    d = np.linalg.norm(x[[0,3,6]]-np.array([x[9],0.,x[12]]))
    std *= noiseDistFactor*d

  current_position_noise = np.zeros(std.shape)
  for i,s in enumerate(std):
    if s > 0:
      current_position_noise[i] = np.random.normal(0., s)
  
  xn = np.zeros(x.shape)
  xn[:] = x[:]
  
  for i,idx in enumerate([0,3,6]):
    if noise_dim == -1 or i != noise_dim:
      xn[ idx ] += current_position_noise[i]
  
#  print current_position_noise
#  print "was, ", x
#  print "NOISE, ", xn
  
  return xn

# ----------------------------------------------------------------------------------------

