from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import Patch, Ellipse
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from scipy.interpolate import spline

import numpy as np
import casadi as ca

__author__ = 'belousov'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class Plotter:
    # ========================================================================
    #                                  2D
    # ========================================================================
    # -------------------------- Helper methods ---------------------------- #
    @staticmethod
    def _create_ellipse(mu, cov):
        if len(mu) != 2 and cov.shape != (2, 2):
            raise TypeError('Arguments should be 2D')

        s = 6 # 6 -> 95%; 9.21 -> 99%
        w, v = np.linalg.eigh(cov)
        alpha = np.rad2deg(np.arctan2(v[1, 1], v[1, 0]))
        width = 2 * np.sqrt(s * w[1])
        height = 2 * np.sqrt(s * w[0])

        # Create the ellipse
        return Ellipse(mu, width, height, alpha,
                       fill=True, color='y', alpha=0.1)

    @staticmethod
    def _plot_arrows(name, ax, x, y, phi):
        x_vec = ca.cos(phi)
        y_vec = ca.sin(phi)
        ax.quiver(x, y, x_vec, y_vec,
                  units='xy', angles='xy', scale=1, width=0.08,
                  headwidth=4, headlength=6, headaxislength=5,
                  color='r', alpha=0.8, lw=0.1)
        return [Patch(color='red', label=name)]

    @staticmethod
    def _plot_arrows_3D(name, ax, x, y, phi, psi):
        x = ca.veccat(x)
        y = ca.veccat(y)
        z = ca.DMatrix.zeros(x.size())
        phi = ca.veccat(phi)
        psi = ca.veccat(psi)
        x_vec = ca.cos(psi) * ca.cos(phi)
        y_vec = ca.cos(psi) * ca.sin(phi)
        z_vec = ca.sin(psi)
        ax.quiver(x + x_vec, y + y_vec, z + z_vec,
                  x_vec, y_vec, z_vec,
                  color='r', alpha=0.8)
        return [Patch(color='red', label=name)]

    # ---------------------------- Trajectory ------------------------------ #
    @classmethod
    def plot_trajectory(cls, ax, x_all):
        [catcher_handle] = cls._plot_trajectory("Catcher's trajectory",
                                       ax, x_all, ('x_c', 'y_c'))
        [gaze_handle] = cls._plot_arrows("Catcher's gaze", ax,
                         x_all[:, 'x_c'], x_all[:, 'y_c'], x_all[:, 'phi'])
        [ball_handle] = cls._plot_trajectory('Ball trajectory',
                                    ax, x_all, ('x_b', 'y_b'))
        ax.grid(True)
        return [catcher_handle, gaze_handle, ball_handle]

    @staticmethod
    def _plot_trajectory(name, ax, x_all, (xl, yl)):
        x = x_all[:, xl]
        y = x_all[:, yl]
        return ax.plot(x, y, label=name, lw=0.8, alpha=0.8, color='g',
                       marker='.', markersize=4, fillstyle='none')

    # --------------------------- Observations ----------------------------- #
    @classmethod
    def plot_observed_ball_trajectory(cls, ax, z_all):
        x = z_all[:, 'x_b']
        y = z_all[:, 'y_b']
        return [ax.scatter(x, y, label='Observed ball trajectory',
                           c='m', marker='+', s=60)]

    # ------------------------ Filtered trajectory ------------------------- #
    @classmethod
    def plot_filtered_trajectory(cls, ax, b_all):
        # Plot line
        [mean_handle] = cls._plot_filtered_ball_mean(
            'Belief trajectory, mean', ax, b_all)

        # Plot ellipses
        [cov_handle] = cls._plot_filtered_ball_cov(
            'Belief trajectory, covariance', ax, b_all)

        # Return handles for the legend
        return [mean_handle, cov_handle]

    @staticmethod
    def _plot_filtered_ball_mean(name, ax, b_all):
        x = b_all[:, 'm', 'x_b']
        y = b_all[:, 'm', 'y_b']
        return ax.plot(x, y, label=name, marker='.', color='m',
                       lw=0.8, alpha=0.9)

    @classmethod
    def _plot_filtered_ball_cov(cls, name, ax, b_all):
        for k in range(b_all.shape[1]):
            e = cls._create_ellipse(b_all[k, 'm', ['x_b', 'y_b']],
                                b_all[k, 'S', ['x_b', 'y_b'], ['x_b', 'y_b']])
            #e.set_fill(False)
            #e.set_facecolor('none')
            e.set_color('darkturquoise')
            e.set_edgecolor('k')
            e.set_alpha(0.2)
            e.set_linewidth(0.5)
            e.set_aa(True)
            e.set_antialiased(True)
            e.set_zorder(1)
            ax.add_patch(e)
        # return [Patch(color='cyan', alpha=0.1, label=name)]
        return [Line2D(b_all[0, 'm', 'x_b'], b_all[0, 'm', 'y_b'],
                       label=name, color='white', alpha=0.8,
                       marker='o', markersize=12, lw=0.2,
                       markerfacecolor='darkturquoise',
                       markeredgecolor='black')]

    # ------------------------ Planned trajectory -------------------------- #
    @classmethod
    def plot_plan(cls, ax, eb_all):
        """Complete plan"""
        handles = cls._plot_plan(ax, eb_all, ('x_b', 'y_b'))
        cls._plot_plan(ax, eb_all, ('x_c', 'y_c'))
        handles.extend(
            cls._plot_arrows("Catcher's gaze", ax,
                         eb_all[:, 'm', 'x_c'],
                         eb_all[:, 'm', 'y_c'],
                         eb_all[:, 'm', 'phi'])
        )
        # Appearance
        ax.grid(True)

        # Return handles
        return handles

    @classmethod
    def _plot_plan(cls, ax, eb_all, (xl, yl)):
        """Plan for one object (ball or catcher)"""
        [plan_m] = cls._plot_plan_m('Plan', ax,
                                    eb_all[:, 'm', xl],
                                    eb_all[:, 'm', yl])
        [plan_S] = cls._plot_plan_S('Posterior', ax,
                                    eb_all[:, 'm', [xl, yl]],
                                    eb_all[:, 'S', [xl, yl], [xl, yl]])
        [plan_L] = cls._plot_plan_L('Prior', ax,
                                    eb_all[:, 'm', [xl, yl]],
                                    eb_all[:, 'L', [xl, yl], [xl, yl]])
        [plan_SL] = cls._plot_plan_SL('Prior + posterior', ax,
                                      eb_all[:, 'm', [xl, yl]],
                                      eb_all[:, 'S', [xl, yl], [xl, yl]],
                                      eb_all[:, 'L', [xl, yl], [xl, yl]])
        return [plan_m, plan_S, plan_L, plan_SL]

    @staticmethod
    def _plot_plan_m(name, ax, x, y):
        """Planned mean"""
        return ax.plot(x, y, label=name, lw=0.7,
                       alpha=0.9, marker='.', color='b')

    @classmethod
    def _plot_plan_S(cls, name, ax, mus, covs):
        """Planned posterior"""
        for k in range(len(mus)):
            e = cls._create_ellipse(mus[k], covs[k])
            e.set_fill(False)
            e.set_color('r')
            e.set_alpha(0.4)
            e.set_lw(1.0)
            ax.add_patch(e)
        return [Line2D(mus[0][0], mus[0][1],
                       label=name, color='white',
                       marker='o', markersize=12,
                       markerfacecolor='white', markeredgecolor='red')]

    @classmethod
    def _plot_plan_L(cls, name, ax, mus, covs):
        """Planned prior"""
        for k in range(len(mus)):
            e = cls._create_ellipse(mus[k], covs[k])
            ax.add_patch(e)
        return [Line2D(mus[0][0], mus[0][1],
                       label=name, color='white', alpha=0.3,
                       marker='o', markersize=12,
                       markerfacecolor='yellow', markeredgecolor='yellow')]

    @classmethod
    def _plot_plan_SL(cls, name, ax, mus, covs, lcovs):
        """Planned prior + posterior"""
        for i in range(len(mus)):
            e = cls._create_ellipse(mus[i], covs[i]+lcovs[i])
            e.set_fill(False)
            e.set_color('g')
            e.set_alpha(0.1)
            e.set_lw(1.0)
            ax.add_patch(e)
        return [Line2D(mus[0][0], mus[0][1],
                       label=name, color='white',
                       marker='o', markersize=12,
                       markerfacecolor='white', markeredgecolor='green')]

    # --------------------- Model predictive control ----------------------- #
    @classmethod
    def plot_mpc(cls, fig, axes, xlim, ylim,
                 model, X_all, Z_all, B_all, EB_all):
        n_delay = model.n_delay
        # Appearance
        axes[0].set_title("Model predictive control, simulation")
        axes[1].set_title("Model predictive control, planning")
        for ax in axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True)
            ax.set_aspect('equal')

        # Plot the first piece
        head = 0
        x_piece = model.x.repeated(X_all[:, head:head+n_delay+1])
        z_piece = model.z.repeated(Z_all[:, head:head+n_delay+1])
        b_piece = model.b.repeated(B_all[:, head:head+n_delay+1])
        handles = cls.plot_trajectory(axes[0], x_piece)
        handles.extend(cls.plot_observed_ball_trajectory(axes[0], z_piece))
        handles.extend(cls.plot_filtered_trajectory(axes[0], b_piece))
        axes[0].legend(handles=handles, loc='upper left')
        fig.canvas.draw()

        # Advance time
        head += n_delay

        # Plot the rest
        for k, _ in enumerate(EB_all):
            # Clear old plan
            axes[1].clear()
            axes[1].set_title("Model predictive control, planning")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            axes[1].grid(True)
            axes[1].set_aspect('equal')

            # Show new plan
            plt.waitforbuttonpress()
            handles = cls.plot_plan(axes[1], EB_all[k][0])
            axes[1].legend(handles=handles, loc='upper left')
            fig.canvas.draw()
            plt.waitforbuttonpress()
            cls.plot_plan(axes[1], EB_all[k][1])
            fig.canvas.draw()

            # Simulate one step
            x_piece = model.x.repeated(X_all[:, head:head+2])
            z_piece = model.z.repeated(Z_all[:, head:head+2])
            b_piece = model.b.repeated(B_all[:, head:head+2])
            plt.waitforbuttonpress()
            cls.plot_trajectory(axes[0], x_piece)
            cls.plot_observed_ball_trajectory(axes[0], z_piece)
            cls.plot_filtered_trajectory(axes[0], b_piece)
            fig.canvas.draw()

            # Advance time
            head += 1

    # --------------------------- Heuristics ------------------------------- #
    @staticmethod
    def plot_heuristics(model, x_all, u_all, n_last=2):
        n_interm_points = 301
        n = len(x_all[:])
        t_all = np.linspace(0, (n - 1) * model.dt, n)
        t_all_dense = np.linspace(t_all[0], t_all[-1], n_interm_points)

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # ---------------- Optic acceleration cancellation ----------------- #
        oac = []
        for k in range(n):
            x_b = x_all[k, ca.veccat, ['x_b', 'y_b']]
            x_c = x_all[k, ca.veccat, ['x_c', 'y_c']]
            r_bc_xy = ca.norm_2(x_b - x_c)
            z_b = x_all[k, 'z_b']
            tan_phi = ca.arctan2(z_b, r_bc_xy)
            oac.append(tan_phi)

        # Fit a line for OAC
        fit_oac = np.polyfit(t_all[:-n_last], oac[:-n_last], 1)
        fit_oac_fn = np.poly1d(fit_oac)

        # Plot OAC
        ax[0, 0].plot(t_all[:-n_last], oac[:-n_last],
                      label='Simulation', lw=3)
        ax[0, 0].plot(t_all, fit_oac_fn(t_all), '--k', label='Linear fit')



        # ------------------- Constant bearing angle ----------------------- #
        cba = []
        d = ca.veccat([ca.cos(model.m0['phi']),
                       ca.sin(model.m0['phi'])])
        for k in range(n):
            x_b = x_all[k, ca.veccat, ['x_b', 'y_b']]
            x_c = x_all[k, ca.veccat, ['x_c', 'y_c']]
            r_cb = x_b - x_c
            cos_gamma = ca.mul(d.T, r_cb) / ca.norm_2(r_cb)
            cba.append(np.rad2deg(np.float(ca.arccos(cos_gamma))))

        # Fit a const for CBA
        fit_cba = np.polyfit(t_all[:-n_last], cba[:-n_last], 0)
        fit_cba_fn = np.poly1d(fit_cba)

        # Smoothen the trajectory
        t_part_dense = np.linspace(t_all[0], t_all[-n_last-1], 301)
        cba_smooth = spline(t_all[:-n_last], cba[:-n_last], t_part_dense)
        ax[1, 0].plot(t_part_dense, cba_smooth, lw=3, label='Simulation')

        # Plot CBA
        # ax[1, 0].plot(t_all[:-n_last], cba[:-n_last],
        #               label='$\gamma \\approx const$')
        ax[1, 0].plot(t_all, fit_cba_fn(t_all), '--k', label='Constant fit')




        # ---------- Generalized optic acceleration cancellation ----------- #
        goac_smooth = spline(t_all,
                            model.m0['phi'] - x_all[:, 'phi'],
                            t_all_dense)

        n_many_last = n_last *\
                      n_interm_points / (t_all[-1] - t_all[0]) * model.dt
        # Delta
        ax[0, 1].plot(t_all_dense[:-n_many_last],
                      np.rad2deg(goac_smooth[:-n_many_last]), lw=3,
                      label=r'Rotation angle $\delta$')
        # Gamma
        ax[0, 1].plot(t_all[:-n_last], cba[:-n_last], '--', lw=2,
                      label=r'Bearing angle $\gamma$')
        # ax[0, 1].plot([t_all[0], t_all[-1]], [30, 30], 'k--',
        #               label='experimental bound')
        # ax[0, 1].plot([t_all[0], t_all[-1]], [-30, -30], 'k--')
        # ax[0, 1].yaxis.set_ticks(range(-60, 70, 30))



        # Derivative of delta
        # ax0_twin = ax[0, 1].twinx()
        # ax0_twin.step(t_all,
        #               np.rad2deg(np.array(ca.veccat([0, u_all[:, 'w_phi']]))),
        #               'g-', label='derivative $\mathrm{d}\delta/\mathrm{d}t$')
        # ax0_twin.set_ylim(-90, 90)
        # ax0_twin.yaxis.set_ticks(range(-90, 100, 30))
        # ax0_twin.set_ylabel('$\mathrm{d}\delta/\mathrm{d}t$, deg/s')
        # ax0_twin.yaxis.label.set_color('g')
        # ax0_twin.legend(loc='lower right')

        # -------------------- Linear optic trajectory --------------------- #
        lot_beta = []
        x_b = model.m0[ca.veccat, ['x_b', 'y_b']]
        for k in range(n):
            x_c = x_all[k, ca.veccat, ['x_c', 'y_c']]
            d = ca.veccat([ca.cos(x_all[k, 'phi']),
                           ca.sin(x_all[k, 'phi'])])
            r = x_b - x_c
            cos_beta = ca.mul(d.T, r) / ca.norm_2(r)

            beta = ca.arccos(cos_beta)
            tan_beta = ca.tan(beta)
            lot_beta.append(tan_beta)

            # lot_beta.append(np.rad2deg(np.float(ca.arccos(cos_beta))))
        # lot_alpha = np.rad2deg(np.array(x_all[:, 'psi']))

        lot_alpha = ca.tan(x_all[:, 'psi'])

        # Fit a line for LOT
        fit_lot = np.polyfit(lot_alpha[model.n_delay:-n_last],
                             lot_beta[model.n_delay:-n_last], 1)
        fit_lot_fn = np.poly1d(fit_lot)

        # Plot
        ax[1, 1].scatter(lot_alpha[model.n_delay:-n_last],
                         lot_beta[model.n_delay:-n_last],
                         label='Simulation')
        ax[1, 1].plot(lot_alpha[model.n_delay:-n_last],
                      fit_lot_fn(lot_alpha[model.n_delay:-n_last]),
                      '--k', label='Linear fit')

        fig.tight_layout()
        return fig, ax

    # ========================================================================
    #                                3D
    # ========================================================================
    @classmethod
    def plot_trajectory_3D(cls, ax, x_all):
        cls._plot_ball_trajectory_3D('Ball trajectory 3D', ax, x_all)
        cls._plot_catcher_trajectory_3D('Catcher trajectory 3D', ax, x_all)
        cls._plot_arrows_3D('Catcher gaze', ax,
                         x_all[:, 'x_c'], x_all[:, 'y_c'],
                         x_all[:, 'phi'], x_all[:, 'psi'])

    @staticmethod
    def _plot_ball_trajectory_3D(name, ax, x_all):
        return ax.scatter3D(x_all[:, 'x_b'],
                            x_all[:, 'y_b'],
                            x_all[:, 'z_b'],
                            label=name, color='g')

    @staticmethod
    def _plot_catcher_trajectory_3D(name, ax, x_all):
        return ax.scatter3D(x_all[:, 'x_c'],
                            x_all[:, 'y_c'],
                            0,
                            label=name, color='g')
