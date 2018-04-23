from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import csv

from sys import path
path.append(r".../casadi-py27-np1.9.1-v3.0.0")
import casadi as ca

from model import Model
from simulator import Simulator
from planner import Planner
from plotter import Plotter

np.set_printoptions(suppress=True, precision=4)

__author__ = 'belousov'


# ============================================================================
#                              Initialization
# ============================================================================
# Model creation wrapper
def new_model(
        # Initial conditions
        x_b0=0, y_b0=0, z_b0=0, vx_b0=10, vy_b0=5, vz_b0=15,
        x_c0=20, y_c0=5, vx_c0=0, vy_c0=0,
        # Initial covariance
        S0=ca.diagcat([0.1, 0.1, 0, 0.1, 0.1, 0,
                       1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]) * 0.25,
        # Hypercovariance weight
        L0_weight=1e-5,
        # Mass of the ball
        mass=0.15,
        # Discretization time step
        dt=0.1,
        # Number of Runge-Kutta integration intervals per time step
        n_rk=10,
        # Reaction time (in units of dt)
        n_delay=1,
        # System noise weight
        M_weight=1e-3,
        # Observation noise
        N_min=1e-2,  # when looking directly at the ball
        N_max=1e0,   # when the ball is 90 degrees from the gaze direction
        # Final cost: w_cl * distance_between_ball_and_catcher
        w_cl=1e3,
        # Running cost on controls: u.T * R * u
        R=1e0 * ca.diagcat([1e1, 1e0, 1e0, 1e-1]),
        # Final cost of uncertainty: w_Sl * tr(S)
        w_Sl=1e3,
        # Running cost of uncertainty: w_S * tr(S)
        w_S=1e2,
        # Control limits
        F_c1=7.5, F_c2=2.5,
        w_max=2 * ca.pi,
        psi_max=0.8 * ca.pi/2,
):
    phi0 = ca.arctan2(y_b0-y_c0, x_b0-x_c0)  # direction towards the ball
    if phi0 < 0:
        phi0 += 2 * ca.pi
    psi0 = 0

    # Initial mean
    m0 = ca.DMatrix([x_b0, y_b0, z_b0, vx_b0, vy_b0, vz_b0,
                     x_c0, y_c0, vx_c0, vy_c0, phi0, psi0])

    # Hypercovariance
    L0 = ca.DMatrix.eye(m0.size()) * L0_weight

    # System noise matrix
    M = ca.DMatrix.eye(m0.size()) * M_weight

    # Catcher dynamics is less noisy
    M[-6:, -6:] = ca.DMatrix.eye(6) * 1e-5

    return Model((m0, S0, L0), dt, n_rk, n_delay, (M, N_min, N_max),
                 (w_cl, R, w_Sl, w_S), (F_c1, F_c2, w_max, psi_max))


# ============================================================================
#                           Plan single trajectory
# ============================================================================
def one_plan():
    model = new_model()
    plan, lam_x, lam_g = Planner.create_plan(model)
    plan, lam_x,  lam_g = Planner.create_belief_plan(
        model, warm_start=True,
        x0=plan, lam_x0=lam_x, lam_g0=lam_g
    )
    x_all = plan.prefix['X']
    u_all = plan.prefix['U']
    eb_all = Simulator.simulate_eb_trajectory(model, u_all)

    # Plot
    fig, ax = plt.subplots()
    fig.tight_layout()
    handles = Plotter.plot_plan(ax, eb_all)
    ax.legend(handles=handles, loc='upper left')
    ax.set_aspect('equal')
    plt.show()


# ============================================================================
#                         Model predictive control
# ============================================================================
def run_mpc(n_delay=1, M_weight=1e-3):
    # Create models for simulation and planning
    model = new_model(n_delay=n_delay)
    model_p = new_model(n_delay=n_delay, M_weight=M_weight)

    # Run MPC
    X_all, U_all, Z_all, B_all, EB_all = Simulator.mpc(model, model_p)

    # Cast simulation results for ease of use
    x_all = model.x.repeated(X_all)
    u_all = model.u.repeated(U_all)
    z_all = model.z.repeated(Z_all)
    b_all = model.b.repeated(B_all)

    # Plot full simulation
    plot_full(x_all, z_all, b_all)

    # Plot heuristics
    # model = new_model()
    # fig = Plotter.plot_heuristics(model, x_all, u_all)
    # plt.show()

    return X_all, U_all, Z_all, B_all, EB_all, model


# ============================================================================
#                                Plotting
# ============================================================================
def plot_full(x_all, z_all, b_all):
    fig, ax = plt.subplots()
    fig.tight_layout()
    handles = Plotter.plot_trajectory(ax, x_all)
    handles.extend(Plotter.plot_observed_ball_trajectory(ax, z_all))
    handles.extend(Plotter.plot_filtered_trajectory(ax, b_all))
    ax.legend(handles=handles, loc='upper left')
    ax.set_aspect('equal')
    plt.show()

def plot_step_by_step(X_all, U_all, Z_all, B_all, EB_all, model):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.tight_layout()
    xlim = (-5, 35)
    ylim = (-5, 30)
    Plotter.plot_mpc(
        fig, axes, xlim, ylim, model, X_all, Z_all, B_all, EB_all
    )


# ============================================================================
#                                   Body
# ============================================================================
# one_plan()

# stuff = run_mpc()
# plot_step_by_step(*stuff)
if __name__ == "__main__":
  for i in range(1):
      run_mpc(n_delay=1+i, M_weight=10**(-1) * 1e-2)




























