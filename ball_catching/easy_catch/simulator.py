from __future__ import division

import casadi as ca
from planner import Planner

__author__ = 'belousov'


class Simulator:

    # ========================================================================
    #                           True noisy trajectory
    # ========================================================================
    @staticmethod
    def simulate_trajectory(model, u_all):
        xk = model.x0.cat
        x_all = [xk]
        for uk in u_all[:]:
            [xk_next] = model.Fn([xk, uk])
            x_all.append(xk_next)
            xk = xk_next
        x_all = model.x.repeated(ca.horzcat(x_all))
        return x_all

    # ========================================================================
    #                              Observations
    # ========================================================================
    @staticmethod
    def simulate_observed_trajectory(model, x_all):
        z_all = []
        for xk in x_all[:]:
            [zk] = model.hn([xk])
            z_all.append(zk)
        z_all = model.z.repeated(ca.horzcat(z_all))
        return z_all

    # ========================================================================
    #                         Filtered observations
    # ========================================================================
    @staticmethod
    def filter_observed_trajectory(model, z_all, u_all):
        n = len(u_all[:])
        bk = model.b0
        b_all = [bk]
        for k in range(n):
            [bk_next] = model.EKF([bk, u_all[k], z_all[k+1]])
            b_all.append(bk_next)
            bk = bk_next
        b_all = model.b.repeated(ca.horzcat(b_all))
        return b_all

    # ========================================================================
    #                       Extended belief trajectory
    # ========================================================================
    @staticmethod
    def simulate_eb_trajectory(model, u_all):
        ebk = model.eb0
        eb_all = [ebk]
        for uk in u_all[:]:
            [ebk_next] = model.EBF([ebk, uk])
            eb_all.append(ebk_next)
            ebk = ebk_next
        eb_all = model.eb.repeated(ca.horzcat(eb_all))
        return eb_all

    # ========================================================================
    #                       Model predictive control
    # ========================================================================
    @classmethod
    def mpc(cls, model, model_p):
        # cls: simulate first n_delay time-steps with zero controls
        u_all = model.u.repeated(ca.DMatrix.zeros(model.nu, model.n_delay))
        x_all = cls.simulate_trajectory(model, u_all)
        z_all = cls.simulate_observed_trajectory(model, x_all)
        b_all = cls.filter_observed_trajectory(model_p, z_all, u_all)

        # Store simulation results
        X_all = x_all.cast()
        Z_all = z_all.cast()
        U_all = u_all.cast()
        B_all = b_all.cast()

        # Advance time
        model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])

        # Iterate until the ball hits the ground
        EB_all = []
        k = 0  # pointer to current catcher observation (= now - n_delay)
        while model.n != 0:
            # Reaction delay compensation
            eb_all_head = cls.simulate_eb_trajectory(
                model_p,
                model_p.u.repeated(U_all[:, k:k+model_p.n_delay])
            )
            model_p.set_initial_state(
                eb_all_head[-1, 'm'],
                eb_all_head[-1, 'm'],
                eb_all_head[-1, 'L'] + eb_all_head[-1, 'S']
            )
            if model_p.n == 0:
                break

            # Planner: plan for model_p.n time steps
            plan, lam_x, lam_g = Planner.create_plan(model_p)
            # plan, lam_x, lam_g = Planner.create_plan(
            #   model_p, warm_start=True,
            #   x0=plan, lam_x0=lam_x, lam_g0=lam_g
            # )
            belief_plan, _, _ = Planner.create_belief_plan(
                model_p, warm_start=True,
                x0=plan, lam_x0=lam_x, lam_g0=lam_g
            )
            u_all = model_p.u.repeated(ca.horzcat(belief_plan['U']))
            # u_all = model_p.u.repeated(ca.horzcat(plan['U']))

            # cls: simulate ebelief trajectory for plotting
            eb_all_tail = cls.simulate_eb_trajectory(model_p, u_all)

            # cls: execute the first action
            x_all = cls.simulate_trajectory(model, [u_all[0]])
            z_all = cls.simulate_observed_trajectory(model, x_all)
            b_all = cls.filter_observed_trajectory(
                model_p, z_all, [u_all[0]]
            )

            # Save simulation results
            X_all.appendColumns(x_all.cast()[:, 1:])
            Z_all.appendColumns(z_all.cast()[:, 1:])
            U_all.appendColumns(u_all.cast()[:, 0])
            B_all.appendColumns(b_all.cast()[:, 1:])
            EB_all.append([eb_all_head, eb_all_tail])

            # Advance time
            model.set_initial_state(x_all[-1], b_all[-1, 'm'], b_all[-1, 'S'])
            model_p.set_initial_state(
                model_p.b(B_all[:, k+1])['m'],
                model_p.b(B_all[:, k+1])['m'],
                model_p.b(B_all[:, k+1])['S']
            )
            k += 1
        return X_all, U_all, Z_all, B_all, EB_all












