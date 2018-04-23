from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import csv

from sys import path
#path.append(r"/home/belousov/local/casadi-py27-np1.9.1-v2.4.3")
path.append(r"/Users/Hoefer/Workspace/casadi-py27-np1.9.1-v2.4.3")

import casadi as ca

from model import Model
from simulator import Simulator
from planner import Planner
import plotter
from plotter import Plotter

np.set_printoptions(suppress=True, precision=4)

__author__ = 'belousov'


# ============================================================================
#                              Initialization
# ============================================================================
# Initial condition
x_b0 = y_b0 = z_b0 = 0
vx_b0 = 10
vy_b0 = 4
vz_b0 = 15

x_c0 = 22
y_c0 = 7
vx_c0 = vy_c0 = 0
phi0 = ca.arctan2(y_b0-y_c0, x_b0-x_c0)  # direction towards the ball
if phi0 < 0:
    phi0 += 2 * ca.pi
psi0 = 0

# Initial mean
m0 = ca.DMatrix([x_b0, y_b0, z_b0, vx_b0, vy_b0, vz_b0,
                 x_c0, y_c0, vx_c0, vy_c0, phi0, psi0])
# Initial covariance
S0 = ca.diagcat([0.2, 0.2, 0, 0.5, 0.5, 0,
                 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]) * 0.25
# Hypercovariance
L0 = ca.DMatrix.eye(m0.size()) * 1e-5
# Discretization step
dt = 0.1
# Number of Runge-Kutta integration intervals per time step
n_rk = 10
# Reaction time (in units of dt)
n_delay = 2
# System noise matrix
M = ca.DMatrix.eye(m0.size()) * 1e-3
M[-6:, -6:] = ca.DMatrix.eye(6) * 1e-5  # catcher's dynamics is less noisy
# Observation noise
N_min = 1e-2  # when looking directly at the ball
N_max = 1e0   # when the ball is 90 degrees from the gaze direction
# Final cost: w_cl * distance_between_ball_and_catcher
w_cl = 1e3
# Running cost on controls: u.T * R * u
R = 1e0 * ca.diagcat([1e1, 1e0, 1e0, 1e-1])
# Final cost of uncertainty: w_Sl * tr(S)
w_Sl = 1e3
# Running cost of uncertainty: w_S * tr(S)
w_S = 1e2
# Control limits
F_c1, F_c2 = 7.5, 2.5
w_max = 2 * ca.pi
psi_max = 0.8 * ca.pi/2

# Model creation wrapper
def new_model():
    return Model((m0, S0, L0), dt, n_rk, n_delay, (M, N_min, N_max),
                 (w_cl, R, w_Sl, w_S), (F_c1, F_c2, w_max, psi_max))

# Create model
model = new_model()

# Handler for red arrow patches
import matplotlib.patches as mpatches
class AnyObjectHandler():
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height

        patch = mpatches.FancyArrow(0, 0.5 * height, width, 0,
                                length_includes_head=True,
                                head_width=0.75 * height, color='r')
        # patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
        #                            edgecolor='black', hatch='xx', lw=3,
        #                            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch



# ============================================================================
#                             Plan trajectory
# ============================================================================
# Find optimal controls
plan, lam_x, lam_g = Planner.create_plan(model)
# plan, lam_x, lam_g = Planner.create_plan(model, warm_start=True,
#                                          x0=plan, lam_x0=lam_x, lam_g0=lam_g)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Simulate ebelief trajectory
eb_all = Simulator.simulate_eb_trajectory(model, u_all)

# Plot 2D
fig, ax = plt.subplots()
handles = Plotter.plot_plan(ax, eb_all)
ax.legend(handles=handles, loc='upper left')
ax.set_aspect('equal')
fig.tight_layout()

# Plot 3D
# fig_3D = plt.figure(figsize=(10, 10))
# ax_3D = fig_3D.add_subplot(111, projection='3d')
# Plotter.plot_trajectory_3D(ax_3D, x_all)


# ============================================================================
#                           Belief space planning
# ============================================================================
# Find optimal controls
plan, lam_x,  lam_g = Planner.create_belief_plan(model, warm_start=True,
                                  x0=plan, lam_x0=lam_x, lam_g0=lam_g)
x_all = plan.prefix['X']
u_all = plan.prefix['U']

# Simulate ebelief trajectory
eb_all = Simulator.simulate_eb_trajectory(model, u_all)

reload(plotter)
from plotter import Plotter

# Plot 2D
fig, ax = plt.subplots()
handles = Plotter.plot_plan(ax, eb_all)
ax.legend(handles=handles, loc='upper left', fontsize=10,
          handler_map={mpatches.Patch: AnyObjectHandler()})
ax.set_aspect('equal')
ax.set_xlim(-2, 35)
ax.set_xlabel('Distance $x$ [m]', fontsize=10, labelpad=2)
ax.set_ylabel('Distance $y$ [m]', fontsize=10, labelpad=2)
fig.tight_layout()


fig.savefig('plan1.pdf', bbox_inches='tight')

# Plot 3D
# fig_3D = plt.figure(figsize=(10, 10))
# ax_3D = fig_3D.add_subplot(111, projection='3d')
# Plotter.plot_trajectory_3D(ax_3D, x_all)

# ============================================================================
#                   Simulate trajectory and observations
# ============================================================================
# Nominal controls for simulation
# u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, 15))
# u_all[:, 'v'] = 5

# Initial state is drawn from N(m0, S0)
# model.init_x0()

# Simulate
x_all = Simulator.simulate_trajectory(model, u_all)
z_all = Simulator.simulate_observed_trajectory(model, x_all)
b_all = Simulator.filter_observed_trajectory(model, z_all, u_all)

# Plot 2D
fig, ax = plt.subplots(figsize=(10, 10))
fig.tight_layout()
Plotter.plot_trajectory(ax, x_all)
Plotter.plot_observed_ball_trajectory(ax, z_all)
Plotter.plot_filtered_trajectory(ax, b_all)

# Plot 3D
# fig_3D = plt.figure(figsize=(10, 10))
# ax_3D = fig_3D.add_subplot(111, projection='3d')
# Plotter.plot_trajectory_3D(ax_3D, x_all)
#
# plt.show()


# ============================================================================
#                         Model predictive control
# ============================================================================
# ----------------------------- Simulation --------------------------------- #
# Create models for simulation and planning
model = new_model()
model_p = new_model()

# Run MPC
X_all, U_all, Z_all, B_all, EB_all = Simulator.mpc(model, model_p)

# Cast simulation results for ease of use
x_all = model.x.repeated(X_all)
u_all = model.u.repeated(U_all)
z_all = model.z.repeated(Z_all)
b_all = model.b.repeated(B_all)


# ---------------------- Step-by-step plotting ----------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.tight_layout()
xlim = (-5, 35)
ylim = (-5, 20)
Plotter.plot_mpc(fig, axes, xlim, ylim,
                 model, X_all, Z_all, B_all, EB_all)


# -------------------------- Plot full simulation -------------------------- #
# Plot 2D
reload(plotter)
from plotter import Plotter

fig, ax = plt.subplots()
handles = Plotter.plot_trajectory(ax, x_all)
handles.extend(Plotter.plot_observed_ball_trajectory(ax, z_all))
handles.extend(Plotter.plot_filtered_trajectory(ax, b_all))
ax.legend(handles=handles, fontsize=9,
          bbox_to_anchor=(0.92, 0.59), labelspacing=0.2,
          handler_map={mpatches.Patch: AnyObjectHandler()})
ax.set_aspect('equal')
ax.set_xlabel('Distance $x$ [m]', labelpad=2, fontsize=10)
ax.set_ylabel('Distance $y$ [m]', labelpad=-2, fontsize=10)
ax.set_xlim(-1, 35)
ax.set_ylim(-1, 11)
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.tick_params(labelsize=10)
ax.grid(zorder=0)
fig.tight_layout()


fig.savefig('traj2.pdf', bbox_inches='tight')

# Plot 3D
# fig_3D = plt.figure(figsize=(10, 10))
# ax_3D = fig_3D.add_subplot(111, projection='3d')
# Plotter.plot_trajectory_3D(ax_3D, model.x.repeated(X_all))


# ---------------------------- Heuristics ---------------------------------- #
model = new_model()

reload(plotter)
from plotter import Plotter

# start of figure
fig, ax = Plotter.plot_heuristics(model, x_all, u_all)

# OAC
ax[0, 0].set_title('Optic acceleration cancellation (OAC)',
                   fontsize=11, y=1.02)
ax[0, 0].set_xlabel('Time $t$ [sec]', fontsize=10, labelpad=2)
ax[0, 0].set_ylabel(r"Tangent of elevation angle $\tan\alpha$",
                    labelpad=2, fontsize=10)
ax[0, 0].set_xlim(0, 3)
ax[0, 0].set_ylim(0, 1.6)
ax[0, 0].set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
ax[0, 0].tick_params(labelsize=10)
ax[0, 0].grid(True)
ax[0, 0].set_axisbelow(True)
ax[0, 0].legend(loc='lower right', fontsize=10)

# CBA
ax[1, 0].set_title('Constant bearing angle (CBA)',
                   fontsize=11, y=1.02)
ax[1, 0].set_xlabel(r'Time $t$ [sec]', fontsize=10, labelpad=2)
ax[1, 0].set_ylabel(r'Bearing angle $\gamma$ [deg]', fontsize=10, labelpad=2)
ax[1, 0].set_xlim(0, 3)
ax[1, 0].set_ylim(-5, 25)
ax[1, 0].set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
ax[1, 0].tick_params(labelsize=10)
ax[1, 0].grid(True)
ax[1, 0].set_axisbelow(True)
ax[1, 0].legend(loc='lower right', fontsize=10)

# GOAC
ax[0, 1].set_title('Tracking heuristic (part of GOAC)',
                   fontsize=11, y=1.02)
ax[0, 1].set_xlabel(r'Time $t$ [sec]', fontsize=10, labelpad=2)
ax[0, 1].set_ylabel(r'Angle [deg]', fontsize=10, labelpad=2)
ax[0, 1].set_xlim(0, 3)
ax[0, 1].set_ylim(-10, 140)
ax[0, 1].set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
ax[0, 1].tick_params(labelsize=10)
ax[0, 1].grid(True)
ax[0, 1].set_axisbelow(True)
ax[0, 1].legend(loc='upper right', fontsize=10)

# LOT
ax[1, 1].set_title('Linear optical trajectory (LOT)',
                   fontsize=11, y=1.02)
ax[1, 1].set_xlabel(r'Tangent of elevation angle $\tan\alpha$',
                    fontsize=10, labelpad=2)
ax[1, 1].set_ylabel(r'Tangent of hor. opt. angle $\tan\beta$',
                    fontsize=10, labelpad=2)

# Very specific
ax[1, 1].set_xlim(0, 3.5)
ax[1, 1].set_ylim(-30, 8)
ax[1, 1].set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
ax[1, 1].set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
ax[1, 1].tick_params(labelsize=10)
ax[1, 1].grid(True)
ax[1, 1].set_axisbelow(True)
ax[1, 1].legend(loc='lower left', fontsize=10)

# end of figure
fig.tight_layout()


fig.savefig('heur_plan.eps', bbox_inches='tight')


# Pickle
import pickle
file_name = 'traj1'
with open(file_name, 'w') as f:
    pickle.dump([x_all, u_all, z_all, b_all], f)

# Unpickle
with open(file_name) as f:
    x_all, u_all, z_all, b_all = pickle.load(f)



# ============================================================================
#                  Save/load trajectory to/from a csv-file
# ============================================================================
def save_trajectory(x_all, filename):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for k in range(len(x_all[:])):
            writer.writerow(list(x_all[k]))


def load_trajectory(model, filename):
    x_all = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            x_all.append(row)
    return model.x.repeated(ca.DMatrix(np.array(x_all, dtype=np.float64)).T)




















