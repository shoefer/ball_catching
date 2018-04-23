import casadi as ca
import casadi.tools as cat

__author__ = 'belousov'


class Planner:

    # ========================================================================
    #                            Simple planning
    # ========================================================================
    @classmethod
    def create_plan(cls, model, warm_start=False,
                    x0=0, lam_x0=0, lam_g0=0):
        # Degrees of freedom for the optimizer
        V = cat.struct_symSX([
            (
                cat.entry('X', repeat=model.n+1, struct=model.x),
                cat.entry('U', repeat=model.n, struct=model.u)
            )
        ])

        # Box constraints
        [lbx, ubx] = cls._create_box_constraints(model, V)

        # Force the catcher to always look forward
        # lbx['U', :, 'theta'] = ubx['U', :, 'theta'] = 0

        # Non-linear constraints
        [g, lbg, ubg] = cls._create_nonlinear_constraints(model, V)

        # Objective function
        J = cls._create_objective_function(model, V, warm_start)

        # Formulate non-linear problem
        nlp = ca.SXFunction('nlp', ca.nlpIn(x=V), ca.nlpOut(f=J, g=g))
        op = {# Linear solver
              #'linear_solver':              'ma57',
              # Acceptable termination
              'acceptable_iter':            5}

        if warm_start:
            op['warm_start_init_point'] = 'yes'
            op['fixed_variable_treatment'] = 'make_constraint'

        # Initialize solver
        solver = ca.NlpSolver('solver', 'ipopt', nlp, op)

        # Solve
        if warm_start:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg,
                         lam_x0=lam_x0, lam_g0=lam_g0)
        else:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        return V(sol['x']), sol['lam_x'], sol['lam_g']

        # sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        # return V(sol['x']), sol['lam_x'], sol['lam_g']

    @staticmethod
    def _create_nonlinear_constraints(model, V):
        g, lbg, ubg = [], [], []
        for k in range(model.n):
            # Multiple shooting
            [xk_next] = model.F([V['X', k], V['U', k]])
            g.append(xk_next - V['X', k+1])
            lbg.append(ca.DMatrix.zeros(model.nx))
            ubg.append(ca.DMatrix.zeros(model.nx))

            # Control constraints
            constraint_k = model._set_constraint(V, k)
            g.append(constraint_k)
            lbg.append(-ca.inf)
            ubg.append(0)
        g = ca.veccat(g)
        lbg = ca.veccat(lbg)
        ubg = ca.veccat(ubg)
        return [g, lbg, ubg]

    @staticmethod
    def _create_objective_function(model, V, warm_start):
        [final_cost] = model.cl([V['X', model.n]])
        running_cost = 0
        for k in range(model.n):
            [stage_cost] = model.c([V['X', k], V['U', k]])

            # Encourage looking at the ball
            d = ca.veccat([ca.cos(V['X', k, 'psi'])*ca.cos(V['X', k, 'phi']),
                           ca.cos(V['X', k, 'psi'])*ca.sin(V['X', k, 'phi']),
                           ca.sin(V['X', k, 'psi'])])
            r = ca.veccat([V['X', k, 'x_b'] - V['X', k, 'x_c'],
                           V['X', k, 'y_b'] - V['X', k, 'y_c'],
                           V['X', k, 'z_b']])
            r_cos_omega = ca.mul(d.T, r)
            if warm_start:
                cos_omega = r_cos_omega / (ca.norm_2(r) + 1e-6)
                stage_cost += 1e-1 * (1 - cos_omega)
            else:
                stage_cost -= 1e-1 * r_cos_omega * model.dt

            running_cost += stage_cost
        return final_cost + running_cost


    # ========================================================================
    #                            Common functions
    # ========================================================================
    @staticmethod
    def _create_box_constraints(model, V):
        lbx = V(-ca.inf)
        ubx = V(ca.inf)

        # Control limits
        model._set_control_limits(lbx, ubx)

        # State limits
        model._set_state_limits(lbx, ubx)

        # Initial state
        lbx['X', 0] = ubx['X', 0] = model.m0

        return [lbx, ubx]


    # ========================================================================
    #                          Belief space planning
    # ========================================================================
    @classmethod
    def create_belief_plan(cls, model, warm_start=False,
                           x0=0, lam_x0=0, lam_g0=0):
        # Degrees of freedom for the optimizer
        V = cat.struct_symSX([
            (
                cat.entry('X', repeat=model.n+1, struct=model.x),
                cat.entry('U', repeat=model.n, struct=model.u)
            )
        ])

        # Box constraints
        [lbx, ubx] = cls._create_box_constraints(model, V)

        # Non-linear constraints
        [g, lbg, ubg] = cls._create_belief_nonlinear_constraints(model, V)

        # Objective function
        J = cls._create_belief_objective_function(model, V)

        # Formulate non-linear problem
        nlp = ca.SXFunction('nlp', ca.nlpIn(x=V), ca.nlpOut(f=J, g=g))
        op = {# Linear solver
              #'linear_solver':              'ma57',
              # Warm start
              # 'warm_start_init_point':      'yes',
              # Termination
              'max_iter':                   1500,
              'tol':                        1e-6,
              'constr_viol_tol':            1e-5,
              'compl_inf_tol':              1e-4,
              # Acceptable termination
              'acceptable_tol':             1e-3,
              'acceptable_iter':            5,
              'acceptable_obj_change_tol':  1e-2,
              # NLP
              # 'fixed_variable_treatment':   'make_constraint',
              # Quasi-Newton
              'hessian_approximation':      'limited-memory',
              'limited_memory_max_history': 5,
              'limited_memory_max_skipping': 1}

        if warm_start:
            op['warm_start_init_point'] = 'yes'
            op['fixed_variable_treatment'] = 'make_constraint'

        # Initialize solver
        solver = ca.NlpSolver('solver', 'ipopt', nlp, op)

        # Solve
        if warm_start:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg,
                         lam_x0=lam_x0, lam_g0=lam_g0)
        else:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        return V(sol['x']), sol['lam_x'], sol['lam_g']

    @staticmethod
    def _create_belief_nonlinear_constraints(model, V):
        """Non-linear constraints for planning"""
        bk = cat.struct_SX(model.b)
        bk['S'] = model.b0['S']
        g, lbg, ubg = [], [], []
        for k in range(model.n):
            # Belief propagation
            bk['m'] = V['X', k]
            [bk_next] = model.BF([bk, V['U', k]])
            bk_next = model.b(bk_next)

            # Multiple shooting
            g.append(bk_next['m'] - V['X', k+1])
            lbg.append(ca.DMatrix.zeros(model.nx))
            ubg.append(ca.DMatrix.zeros(model.nx))

            # Control constraints
            constraint_k = model._set_constraint(V, k)
            g.append(constraint_k)
            lbg.append(-ca.inf)
            ubg.append(0)

            # Advance time
            bk = bk_next
        g = ca.veccat(g)
        lbg = ca.veccat(lbg)
        ubg = ca.veccat(ubg)
        return [g, lbg, ubg]

    @staticmethod
    def _create_belief_objective_function(model, V):
        # Simple cost
        running_cost = 0
        for k in range(model.n):
            [stage_cost] = model.c([V['X', k], V['U', k]])
            running_cost += stage_cost
        [final_cost] = model.cl([V['X', model.n]])

        # Uncertainty cost
        running_uncertainty_cost = 0
        bk = cat.struct_SX(model.b)
        bk['S'] = model.b0['S']
        for k in range(model.n):
            # Belief propagation
            bk['m'] = V['X', k]
            [bk_next] = model.BF([bk, V['U', k]])
            bk_next = model.b(bk_next)
            # Accumulate cost
            [stage_uncertainty_cost] = model.cS([bk_next])
            running_uncertainty_cost += stage_uncertainty_cost
            # Advance time
            bk = bk_next
        [final_uncertainty_cost] = model.cSl([bk_next])

        return running_cost + final_cost +\
               running_uncertainty_cost + final_uncertainty_cost




















