# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:04:56 2016

@author: shoefer
0: t
1: x
2: y
3: z
4: vx
5: vy
6: vz
"""

import os
import casadi as ca

#from model import Model
from ball_catching.easy_catch.simulator import Simulator
from ball_catching.easy_catch.planner import Planner
from ball_catching.easy_catch import launcher
from ball_catching.dynamics.world import DynamicsModel
from ball_catching.strategies.base import Strategy

import yaml

#import pickle

#from utils import make_hierarchical, make_flat
#from warnings import warn



# ----------------------------------------------------------------------------------------


class MPCStrategy(Strategy):
    control_type = "full"

    def __init__(self, dicts_):
        cls_name = self.__class__.__name__

        # Now handled by run_python
        # self.n_delay = dicts_['Agent/delay']
        # self.n_delay = 0
        # self.n_delay = 1 # FIXME
        self.n_delay = dicts_['%s/delay' % cls_name]

        # Only for debugging:
        self._active = dicts_['%s/active' % cls_name]

        # Optimization parameters
        self.F_c1 = dicts_['%s/F_c1' % cls_name]
        self.F_c2 = dicts_['%s/F_c2' % cls_name]

        self.M = dicts_['%s/M' % cls_name]
        self.N = dicts_['%s/N' % cls_name]

        try:
            self.N_min = dicts_['%s/N_min' % cls_name]
        except:
            self.N_min = self.N

        try:
            self.N_max = dicts_['%s/N_max' % cls_name]
        except:
            self.N_max = self.N

        self.model_kwargs = {'dt': DynamicsModel().DT,
                             'F_c1': self.F_c1,
                             'F_c2': self.F_c2,
                             'n_delay': self.n_delay,
                             # observation noise at center and border equal (default values)
                             'N_min': self.N_min,  # observation noise when looking directly at the ball
                             'N_max': self.N_max,  # observation noise when ball is 90 degrees
                             'M_weight': self.M,  # system covariance noise
                             }

        try:
            self.w_max = dicts_['%s/w_max' % cls_name]
        except:
            self.w_max = None
        if self.w_max is not None:
            self.model_kwargs['w_max'] = self.w_max

        try:
            self.psi_max = dicts_['%s/psi_max' % cls_name]
        except:
            self.psi_max = None
        if self.psi_max is not None:
            self.model_kwargs['psi_max'] = self.psi_max

        # self.uncertainty = dicts_['%s/uncertainty' % cls_name]
        # print ("Uncertainty " + self.uncertainty )
        # if self.uncertainty == "low":
        #   self.model_kwargs.update({
        #            # basically no noise
        #            'M_weight': 1e-15,
        #            'N_min': 1e-15,
        #            'N_max': 1e-15,
        #   })
        # else:
        #   # using default values
        #   pass

        self.log_root = None

    def start(self, **kwargs):
        self._model_initialized = False
        self._current_step = 0
        self._current_ec_step = 0

        self._Xraw = []
        self._Uraw = []

    def _init_model(self, x0):
        print ("Model parameters")
        print (self.model_kwargs)

        self.model_kwargs.update({
            'x_b0': x0[0],
            'y_b0': x0[6],
            'z_b0': x0[3],  # z is up
            'vx_b0': x0[1],
            'vy_b0': x0[7],
            'vz_b0': x0[4],  # z is up
            'x_c0': x0[9],
            'y_c0': x0[12],
            'vx_c0': x0[10],
            'vy_c0': x0[13],
        })

        print ("Building simulation model...")
        self.model = launcher.new_model(**self.model_kwargs)
        print ("Building prediction model...")
        self.model_p = launcher.new_model(**self.model_kwargs)
        print ("Done")

        self._model_initialized = True

        # cls: simulate first n_delay time-steps with zero controls
        if self.n_delay > 0:
            u_all = self.model.u.repeated(ca.DMatrix.zeros(self.model.nu, self.model.n_delay))
            x_all = Simulator.simulate_trajectory(self.model, u_all)
            z_all = Simulator.simulate_observed_trajectory(self.model, x_all)
            b_all = Simulator.filter_observed_trajectory(self.model_p, z_all, u_all)

        else:
            u_all = []
            x_all = self.model.x.repeated(ca.horzcat([self.model.x0.cat]))
            z_all = Simulator.simulate_observed_trajectory(self.model, x_all)
            b_all = self.model.b.repeated(ca.horzcat([self.model.b0]))

        # print x_all.cast()
        # x_all = self._update_X(self.model.n_delay, x_all)
        x_all = self._update_X(x_all, x0)
        # print x_all.cast()
        # z_all = self._update_Z(self.model.n_delay, z_all)
        z_all = self._update_Z(z_all, x0)
        # print z_all.cast()

        # Store simulation results
        self.X_all = x_all.cast()
        self.Z_all = z_all.cast()
        if self.n_delay > 0:
            self.U_all = u_all.cast()
        else:
            self.U_all = None
        self.B_all = b_all.cast()

        # Advance time
        self.model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])

        for i in range(self.n_delay):
            self._Uraw.append(u_all[i])

        self.u_old = u_all

    def write_logs(self, log_root, trial):
        np.savetxt(os.path.join(log_root, "X_raw_%d.txt" % trial),
                   np.asarray([np.asarray(x) for x in self._Xraw]))

        np.savetxt(os.path.join(log_root, "U_raw_%d.txt" % trial),
                   np.asarray([np.asarray(u) for u in self._Uraw]))

        try:
            with open(os.path.join(log_root, "mpc_settings.yml" % trial), "w") as f:
                model_kwargs = {}
                for k, v in self.model_kwargs.items():
                    try:
                        model_kwargs[k] = str(v)
                    except Exception as e:
                        print(e)
                        pass
                yaml.dump(model_kwargs, f)
        except Exception as e:
            print(e)

        try:
            print (self.model.__dict__)
            with open(os.path.join(log_root, "mpc_internal_settings.yml" % trial), "w") as f:
                model_dict = {}
                for k, v in self.model.__dict__.items():
                    try:
                        model_dict[k] = str(v)
                    except Exception as e:
                        print(e)
                        pass
                yaml.dump(model_dict, f)
        except Exception as e:
            print(e)
            pass

    def step(self, i, x, dicts):
        if not self._model_initialized:
            assert (i == 0)
            self._init_model(x)

        # ----
        # n_delay still active?
        if i < self.n_delay:
            self.control_type = "acceleration"
            return [0., 0.]

        self.control_type = "full"

        # ---------------------
        # If more than one time step has passed, we need to "after-simulate"
        if i > 0:

            k = i - self.n_delay

            # HACKY
            if self.n_delay == 0:
                k -= 1

            # get previous action
            u_all = self.u_old

            # PERFORM PREVIOUS SIMULATION STEP
            x_all = Simulator.simulate_trajectory(self.model, [u_all[0]])
            # UPDATE INTERNAL SIMULATOR STATE WITH REAL OBSERVATION
            x_all = self._update_X(x_all, x)

            # PERFORM PREVIOUS OBSERVATION STEP
            z_all = Simulator.simulate_observed_trajectory(self.model, x_all)
            # UPDATE INTERNAL SIMULATOR STATE WITH REAL OBSERVATION
            z_all = self._update_Z(z_all, x)

            b_all = Simulator.filter_observed_trajectory(
                self.model_p, z_all, [u_all[0]]
            )

            # Save simulation results
            self.X_all.appendColumns(x_all.cast()[:, 1:])
            self.Z_all.appendColumns(z_all.cast()[:, 1:])
            if self.U_all is None:
                self.U_all = u_all.cast()[:, 0]
            else:
                self.U_all.appendColumns(u_all.cast()[:, 0])
            self.B_all.appendColumns(b_all.cast()[:, 1:])

            # Advance time
            self.model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])
            self.model_p.set_initial_state(
                self.model_p.b(self.B_all[:, k + 1])['m'],
                self.model_p.b(self.B_all[:, k + 1])['m'],
                self.model_p.b(self.B_all[:, k + 1])['S']
            )

            # Reaction delay compensation
            eb_all_head = Simulator.simulate_eb_trajectory(
                self.model_p,
                self.model_p.u.repeated(self.U_all[:, k:k + self.model_p.n_delay])
            )

        else:
            eb_all_head = self.model_p.eb.repeated(ca.horzcat([self.model_p.eb0]))

        # ---------------------
        # Set initial state of the model
        self.model_p.set_initial_state(
            eb_all_head[-1, 'm'],
            eb_all_head[-1, 'm'],
            eb_all_head[-1, 'L'] + eb_all_head[-1, 'S']
        )

        # Should not happen
        if self.model_p.n <= 0:
            print ("Prediction model will not predict anymore")
            self.control_type = "acceleration"
            return [0., 0.]

        # ---------------------
        # Plan new action

        if self._active:
            try:
                plan, lam_x, lam_g = Planner.create_plan(self.model_p)
                # plan, lam_x, lam_g = Planner.create_plan(
                #   self.model_p, warm_start=True,
                #   x0=plan, lam_x0=lam_x, lam_g0=lam_g
                # )
                belief_plan, _, _ = Planner.create_belief_plan(
                    self.model_p, warm_start=True,
                    x0=plan, lam_x0=lam_x, lam_g0=lam_g
                )
                u_all = self.model_p.u.repeated(ca.horzcat(belief_plan['U']))
            except Exception as e:
                print (e)
                u_all = self.model.u.repeated(ca.DMatrix.zeros(self.model.nu, self.model.n))
        else:
            u_all = self.model.u.repeated(ca.DMatrix.zeros(self.model.nu, self.model.n))

        # now carry out one simulation step and return the resulting state
        x_all = Simulator.simulate_trajectory(self.model, [u_all[0]])

        # we return position, velocity, acceleration
        u = map(float, [
            x_all[1][6],
            x_all[1][8],
            0.,
            x_all[1][7],
            x_all[1][9],
            0.,
        ])
        if DynamicsModel().dim == 2:
            # in 2D setting ignore motion in z
            u = np.asarray(u)
            u[3:] = 0.

        self._Uraw.append(u_all[0])
        self.u_old = u_all

        self._Xraw.append(x_all[0])

        return u

    #  @classmethod
    #  def _transform_U(cls, x_mpc, u_mpc):
    #    [x_b, y_b, z_b, vx_b, vy_b, vz_b, x_c, y_c, vx_c, vy_c, phi, psi] = x_mpc
    #    [F_c, _, _, theta] = u_mpc
    #
    #    return map (float, [
    #      F_c*ca.cos(phi + theta), # FIXME air resistance for agent?
    #      F_c*ca.sin(phi + theta) # FIXME air resistance for agent?
    #    ])

    @classmethod
    def _x_to_xbc(cls, x):
        """
        Convert an easy
        """
        x_bc = np.zeros((15,))
        x_bc[[0, 6, 3]] = map(float, [x[i] for i in [0, 1, 2]])  # position ball
        x_bc[[1, 7, 4]] = map(float, [x[i] for i in [3, 4, 5]])  # velocity ball
        # x_bc[[2,5,8]] = (x_bc[[1,4,7]]-x_bc[[1,4,7]])/self.dt  # acceleration ball
        x_bc[5] = -DynamicsModel().GRAVITY
        x_bc[[9, 12]] = map(float, [x[i] for i in [6, 7]])  # position agent
        x_bc[[10, 13]] = map(float, [x[i] for i in [8, 9, ]])  # velocity agent
        # x_bc[[11,44]] = (x_bc[1:, [10,13]]-x_bc[:-1, [10,13]])/self.dt  # acceleration agent

        return x_bc

    def _update_X(self, X, x):
        """
           X is the CasADi matrix with the entire history of states
           x is the current state
        """
        X_ = X.cast()

        # Ball
        x_cols = [0, 6, 3, 1, 7, 4]
        for i, col in enumerate(x_cols):
            X_[i, -1] = x[col]

        # steps = X_.shape[1]
        #        for t in range(steps):
        #          for i, col in enumerate(x_cols):
        #              X_[i,-t-1] = x[col]
        return self.model.x.repeated(X_)

    def _update_Z(self, Z, x):
        """
           Z is the CasADi matrix with the entire history of observations
           x is the current state
        """
        Z_ = Z.cast()

        z_cols = [0, 6, 3, ]

        for i, col in enumerate(z_cols):
            Z_[i, -1] = x[col]

        # steps = Z_.shape[1]
        #        for t in range(steps):
        #          for i, col in enumerate(z_cols):
        #              Z_[i,-t-1] = x[col]

        return self.model.z.repeated(Z_)


# ---------------------------------------------------------------------------------

from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Patch, Ellipse
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from scipy.interpolate import spline

import numpy as np


class MPCPlotter:
    @staticmethod
    def _plot_arrows(name, ax, x, y, phi):
        r = 2.
        x_vec = r * np.cos(phi)
        y_vec = r * np.sin(phi)
        ax.quiver(x, y, x_vec, y_vec,
                  units='xy', angles='xy', scale=1, width=0.08,
                  headwidth=4, headlength=6, headaxislength=5,
                  color='r', alpha=0.8, lw=0.1)
        return [Patch(color='red', label=name)]

    # ---------------------------- Trajectory ------------------------------ #
    @classmethod
    def plot_trajectory(cls, ax, x_all):
        [catcher_handle] = cls._plot_trajectory("Catcher's trajectory",
                                                ax, x_all, (6, 7), "b")
        [gaze_handle] = cls._plot_arrows("Catcher's gaze", ax,
                                         x_all[:, 6], x_all[:, 7], x_all[:, 10])
        [ball_handle] = cls._plot_trajectory('Ball trajectory',
                                             ax, x_all, (0, 1), "r")
        ax.grid(True)
        return [catcher_handle, gaze_handle, ball_handle]

    @staticmethod
    def _plot_trajectory(name, ax, x_all, (xl, yl), c="g"):
        x = x_all[:, xl]
        y = x_all[:, yl]
        return ax.plot(x, y, label=name, lw=1.8, alpha=0.8, color=c,
                       marker='.', markersize=4, fillstyle='none')