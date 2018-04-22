# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:29:13 2016

@author: Hoefer
"""

import argparse
import os
import sys
import yaml

import numpy as np
import matplotlib.pylab as plt

from ball_catching.config import settings_root
from ball_catching.dynamics.world import DynamicsModel

def dot3(A,B,C):
    """ Returns the matrix multiplication of A*B*C"""
    return np.dot(A, np.dot(B,C))

# ====================================================================

class LQR:  
    def __init__(self):
        pass
        
    @property
    def name(self):
        return "LQR"
        
#    def solve(self, N, A=None, B=None):
#        """Solve the LQR problem, iterating over N steps"""
#        
#        Q, R, P0 = self.Q, self.R, self.H
#        
#        if A is None:
#            A = dynamics.Adt
#        if B is None:
#            B = dynamics.Bdt
#        
#        P = P0[:,:]
#        Plog = [P0]
#        Flog = []
#        
#        for i in range(N):
#            try:
#              F = - np.dot ( np.linalg.inv(R + np.dot(np.dot(B.T, P), B)),
#                          np.dot( np.dot(B.T, P), A ) )
#            except np.linalg.linalg.LinAlgError as e:
#              print "warn: %s" % str(e)
#              F = np.zeros(B.T.shape)
#              
#            Flog.append(F)    
#            P = np.dot ( np.dot( (A + np.dot(B, F)).T, P ),
#                              (A + np.dot(B, F))) + np.dot( np.dot(F.T, R), F) + Q
#            Plog.append(P)
#            
#        self.Plog = Plog
#        self.Flog = Flog
#        
#        return Plog, Flog

    def solve(self, N, dynamics, A=None, B=None, c=None):
        """Solve the LQR problem, iterating over N steps"""
        self.dynamics_local = dynamics
        
        dt = dynamics.DT
        
        Q, R, S = self.Q, self.R, self.H
        D_s, D_a = self.Q.shape[0], self.R.shape[0]
        #P = np.zeros( (D_a, D_s))        
        s = np.zeros( (D_s,) )
        
        if A is None:
            A = dynamics.Adt
        if B is None:
            B = dynamics.Bdt
        if c is None:
            c = dynamics.cdt
            #c = np.zeros( (D_s,) )
        
        F = np.zeros( (N, D_a, D_s) )
        f = np.zeros( (N, D_a) )
        
        inv = np.linalg.inv 
        
        for t in reversed(range(N)):
            C = dot3(B.T, S, A) #+ P
            D = dot3(A.T, S, A) + Q
            E = dot3(B.T, S, B) + R
            d = np.dot(A.T, s+S.dot(c)) #+ q
            e = np.dot(B.T, s+S.dot(c)) #+ r
            #F[t] = - inv(E).dot(C)
            #f[t] = - inv(E).dot(e)
            #S = D + C.T.dot(F[t])
            #s = d + C.T.dot(f[t])

            idx = N-t-1
            F[idx] = - inv(E).dot(C)
            f[idx] = - inv(E).dot(e)
            S = D + C.T.dot(F[idx])
            s = d + C.T.dot(f[idx])
            
        self.F = F
        self.f = f
        
        self.tti = [ i*dt for i in range(N) ]        
        
        return self.tti, self.F, self.f

    def cost_t(self, x, u):
        Q, R, = self.Q, self.R
        x = x.reshape( (-1,1) )
        u = u.reshape( (-1,1) )
        sx = np.dot(x.T, np.dot(Q, x))
        su = np.dot(u.T, np.dot(R, u))
        
        return (sx + su)[0,0]
        
    def cost_final(self, x):
        x = x.reshape( (-1,1) )
        return np.dot(x.T, np.dot(self.H, x))[0,0]

    def J(self, x, u, N):
        """Compute the total cost of a trajectory
           x & u for N steps"""
        Q, R, H = self.Q, self.R, self.H
        
        sum = 0
        for i in range(N-1):
            #FIXME use cost_t
            xx = x[i,:].T
            uu = u[i,:].T
            sx = np.dot(xx.T, np.dot(Q, xx))
            su = np.dot(uu.T, np.dot(R, uu))
            sum += sx
            sum += su
    
        # last step:        
        if x.shape[0] == N:
            #FIXME use cost_final
            sum += np.dot(x[-1,:].T, np.dot(H, (x[-1,:])))
            
        return 0.5 * sum
        
        
# ====================================================================

class iLQR(LQR):  
  
    @property
    def name(self):
        return "iLQR"
  
    def solve(self, N, dynamics, x0=None, u0=None, max_iter=1000, A=None, B=None, c=None, verbose=True):
        """Solve the iLQR problem, iterating over N steps"""
        inv = np.linalg.inv 
  
        self.dynamics_local = dynamics
        dt = dynamics.DT  
  
        # cost matrices
        Q, R, S = self.Q, self.R, self.H
        D_s, D_a = self.Q.shape[0], self.R.shape[0]
        #S = np.zeros( (D_a, D_s))        
        s = np.zeros( (D_s,) )

        if A is None:
            A = dynamics.Adt
        if B is None:
            B = dynamics.Bdt
        if c is None:
            c = dynamics.cdt
            #c = np.zeros( (D_s,) )
  
        g = lambda x,u: dynamics.step(x,u,noise=False)

        if x0 is None:
            x0 = np.zeros( (D_s,) )
            
        if u0 is None:
            u0 = np.zeros( (D_a,) )
          
        tf, N, _, _ = dynamics.get_time_to_impact(x0)
        # initialize state and action matrices
        F = np.zeros( (N, D_a, D_s) )
        f = np.zeros( (N, D_a) )

        # initialize state and action matrices
        x_hat = np.zeros((N+1, D_s)) 
        x_hat_new = np.zeros((N+1, D_s)) 
        u_hat = np.zeros((N, D_a)) 
        u_hat_new = np.zeros((N, D_a)) 
      
        old_cost = np.inf
        
        new_cost = 0.
        
        for opt_iter in range(max_iter):
            alpha = 1.  # line search parameter
            
            # ------------
            # Forward pass
            
            # line search
            first_round = True
            while first_round or (new_cost >= old_cost and np.abs((old_cost - new_cost) / new_cost) >= 1e-4):
              
                first_round = False
                new_cost  = 0.
                
                # initialize trajectory
                x_hat_new[0,:] = x0
                for t in range(N):
                    idx = N-t-1
                  
                    # line search for choosing optimal combination of old and new action
                    u_hat_new[t,:] = (1.0 - alpha)*u_hat[t,:] \
                        + F[idx].dot(x_hat_new[t,:] - (1.0 - alpha)*x_hat[t,:]) + alpha*f[idx]
                    # next time-step
                    x_hat_new[t+1,:] = g(x_hat_new[t,:], u_hat_new[t,:])
  
                    new_cost += self.cost_t(x_hat_new[t,:], u_hat_new[t,:])
  
                new_cost += self.cost_final(x_hat_new[t,:])

                alpha *= 0.5
        
            x_hat[:] = x_hat_new[:]
            u_hat[:] = u_hat_new[:]
        
            if verbose:
                print ("Iter: %d, Alpha: %f, Rel. progress: %f, Cost: %f" % \
                  (opt_iter, (2*alpha), ((old_cost-new_cost)/new_cost), new_cost,))
            
            if np.abs((old_cost - new_cost) / new_cost) < 1e-4:
                break
        
            old_cost = new_cost

            # ------------
            # backward pass        

            # for quadratizing final cost (not implemented)
            #S = np.zeros( (D_s, D_s) )
            #s = np.zeros( (D_s, ) )

            S = self.H
            s = np.zeros( (D_s, ) )
        
            #for (size_t t = ell-1; t != -1; --t) {
            for t in reversed(range(N)):
                # jacobian
                A = dynamics.compute_J(x_hat[t], u_hat[t])
                B = dynamics.Bdt # FIXME nonlinear motion model support
                c = x_hat[t+1] - (A.dot(x_hat[t]) - B.dot(u_hat[t])).flatten()

                C = dot3(B.T, S, A) #+ P
                D = dot3(A.T, S, A) + Q
                E = dot3(B.T, S, B) + R
                d = np.dot(A.T, s+S.dot(c)) #+ q
                e = np.dot(B.T, s+S.dot(c)) #+ r
#                F[t] = - inv(E).dot(C)
#                f[t] = - inv(E).dot(e)                
#                S = D + C.T.dot(F[t])
#                s = d + C.T.dot(f[t])            

                idx = N-t-1
                F[idx] = - inv(E).dot(C)
                f[idx] = - inv(E).dot(e)
                S = D + C.T.dot(F[idx])
                s = d + C.T.dot(f[idx])

        self.F = F
        self.f = f

        self.tti = [ i*dt for i in range(N) ]        
            
        # old style        
        #self.Flog = [ F[t] for t in range(F.shape[0]) ]
      
        return self.tti, self.F, self.f
            
# ====================================================================
        
class SOCBallCatching:

    def __init__(self, solver, dynamics_local,
                 terminal_distance, terminal_velocity, control_effort):
        """ 
          Generates cost matrices Q, H, R and 
          assigns them to the solver (LQR or iLQR)
        """
        
        self.terminal_distance = terminal_distance
        self.terminal_velocity = terminal_velocity
        self.control_effort = control_effort

        D_s = dynamics_local.state_dim
        D_a = dynamics_local.action_dim
        
        Q = np.zeros((D_s,D_s))
        R = np.identity(D_a)*control_effort
        H = np.zeros((D_s,D_s))
        
        # agent terminal_distance to ball (x dimension)
        H[0,0] = terminal_distance
        H[9,9] = terminal_distance
        H[0,9] = H[9,0] = -terminal_distance
        
        # agent terminal_distance to ball (z dimension)
        H[6,6] = terminal_distance
        H[12,12] = terminal_distance
        H[6,12] = H[12,6] = -terminal_distance
        
        # agent velocity at contact 
        H[10,10] = terminal_velocity
        H[13,13] = terminal_velocity
        
        # init solver cost
        solver.Q = Q
        solver.R = R
        solver.H = H
        
        self.dynamics_global = DynamicsModel()

        self.solver = solver
        self.solver.dynamics_local = dynamics_local
        
    def solve(self, N=None):
        fr, dt, dim =  self.dynamics_global.FRAMERATE,\
                       self.dynamics_global.DT,\
                       self.dynamics_global.dim
      
        if N is None:
            N = int(10*fr)  # 10 seconds at current framerate
            
        # --------
        # Use LQR
        if self.solver.name == "LQR":
            #dynamics_local = DynamicsModel(dt=dt, dim=dim, 
            #                               drag=False, copy=True)                  
            ret = self.solver.solve(N, self.solver.dynamics_local)

        # --------
        # Use iLQR
        elif self.solver.name == "iLQR":
            #dynamics_local = DynamicsModel(dt=dt, dim=dim, 
            #                               drag=True, copy=True)                  
            D_s = self.dynamics_global.state_dim
            D_a = self.dynamics_global.action_dim

            # we need to set x0 and u0
            x0 = np.zeros( (D_s,) )

            # using 'far' setting
            x0[1] = 150. # ball velocity x
            #x0[4] = 15.556 # ball velocity z
            x0[4] = 150. # ball velocity z
            #if dim==3:
            #  x0[7] = x0[1] # ball velocity x, y and z
            x0[5] = -self.dynamics_global.GRAVITY
            x0[9] = 300 # agent x-position
            if dim==3:
              #x0[12] = x0[9] # agent z-position
              x0[12] = 30. # agent z-position

            u0 = np.zeros( (D_a,) )
            
            ret = self.solver.solve(N, self.solver.dynamics_local, x0=x0, u0=u0)

        self.tti, self.F, self.f = ret

        return ret
            
        

    def compute_tti(self, x):
        """
          Returns time to impact in seconds
        """
        return self.dynamics_global.get_time_to_impact(x)[0]

    def run(self, x0):
        """ Execute the controller for a given system
            The controller is given through the gain matrices F.
            
            x and u are out variables that will contain the state
            and action trajectories
        """
        dt = self.dynamics_global.DT
        framerate = self.dynamics_global.FRAMERATE
        
        # print some analytical information about example      
        t_n = self.compute_tti(x0)
        print "At what time t is ball on ground:  %f" % (t_n)
        tdt_n = t_n/dt
        tf = int(round (tdt_n))
        print "After how many steps (according to dt) N is ball on ground:  %f" % (tdt_n, )
        x_n = map (lambda t: x0[1,0]*t + x0[0,0], (t_n, ))
        print "At what COORDINATE x  is ball on ground:  %f" % x_n[0]
        z_n = map (lambda t: x0[7,0]*t + x0[6,0], (t_n, ))
        print "At what COORDINATE z  is ball on ground:  %f" % z_n[0]
        
        N_example = tf # this will be used to show example 
  
        # logging
        x = np.zeros( (N_example+1,D_s) )
        u = np.zeros( (N_example,D_a) )
        t = np.arange(0, x.shape[0]+1, dt)
        t = t[:x.shape[0]]
        
        x[0, :] = x0.T
        
        for i in range(1,N_example+1):
            # compute optimal control gains
            tti  = self.compute_tti(x[i-1,:])
            step = int(round(tti*framerate))-1
            F = self.solver.F[step]
            f = self.solver.f[step]
                                
            u[i-1,:] = np.dot(F, x[i-1,:].T ).T
            u[i-1,:] += f.reshape(u[i-1].shape)
    
            # if Flog is None, assume u to be an input variable        
            x[i,:] = self.dynamics_global.step(x[i-1:i,:].T, u[i-1:i,:].T, noise=False).reshape((-1,))
            t[i] = i*dt
          
        return t, x, u,
        
    def build_gain_folder_name(self, output_root=None):
        if output_root is None:
            output_root = os.path.join(settings_root, self.solver.name)
            
        if self.terminal_velocity > 0.:
          output_root += "V%d" % self.terminal_velocity
        output_root += "Dim%d" % self.dynamics_global.dim
        
        output_root += "M%.2f" % self.solver.dynamics_local.mass
        output_root += "R%.2f" % self.solver.dynamics_local.r
        output_root += "C%.2f" % self.solver.dynamics_local.c
        
        framerate = self.dynamics_global.FRAMERATE
        fr_folder = "%.1f" % framerate
      
        cur_output_folder = os.path.join(output_root, fr_folder)
                
        return cur_output_folder

    @staticmethod 
    def _read_gains_from_folder(path):
      """
        Returns a list of tuples (tti, gains)
      """
      g = []
      for _p in os.listdir(path):
        p = os.path.join(path, _p)
        if os.path.splitext(_p)[-1] in [".txt", ".yml", ".pdf"] \
          or os.path.isdir(p):
          continue
        
        g.append ((float(_p), np.loadtxt(p)))
      return g


    def load(self):
        gain_root = self.build_gain_folder_name()
        if not os.path.exists(gain_root):
            print ( "[SOCBallCatching] Solving %s..." % self.solver.name )
            res = self.solve()
            self.export()
            return res

        print ( "[SOCBallCatching] Loading gains from  %s" % gain_root)
            
        # LOAD multiplicative (L) and constant gains (l) from file
        F = []
        f = []
        
        yml_path = os.path.join(gain_root, "cost_info.yml")
        with open(yml_path, "r") as yf:
            y = yaml.load(yf)
            for v in ["terminal_distance", "terminal_velocity", "control_effort"]:
              assert self.__dict__[v] == y[v], \
                "%s has different values. params:%f != yaml:%f" % (v, self.__dict__[v], y[v]) 
        
        for _f in os.listdir(gain_root):
          if _f == "Fm":
            F = self._read_gains_from_folder(os.path.join(gain_root, _f))
          elif _f == "fc":
            f = self._read_gains_from_folder(os.path.join(gain_root, _f))
            
        if len(f) == 0:
          print ("[LQRStrategy] WARN: f is empty")
          f = np.zeros( (len(F), self.dynamics_global.action_dim ))

        # sort by tti
        F = sorted(F, key=lambda x: x[0])
        f = sorted(f, key=lambda x: x[0])
            
        self.tti = [ _f[0] for _f in F]
        self.F = np.asarray( [ _f[1] for _f in F] )
        self.f = np.asarray( [ _f[1] for _f in f] )

        return self.tti, self.F, self.f
        
    def export(self, output_root=None):
        cur_output_folder = self.build_gain_folder_name(output_root)
        framerate = self.dynamics_global.FRAMERATE
      
        # prepare
        import shutil
        try:
          shutil.rmtree(cur_output_folder)
        except:
          pass # don't care
        os.makedirs(cur_output_folder)        

        print "Writing to %s" % cur_output_folder
        
        # save system matrices
        print "Writing system matrices"
        np.savetxt(os.path.join(cur_output_folder, "A.txt"), self.dynamics_global.Adt)
        np.savetxt(os.path.join(cur_output_folder, "B.txt"), self.dynamics_global.Bdt)
        np.savetxt(os.path.join(cur_output_folder, "C.txt"), self.dynamics_global.Cdt)

        # write matrices
        print ("Writing gain matrices")
        
        os.makedirs(os.path.join(cur_output_folder, "Fm"))
        os.makedirs(os.path.join(cur_output_folder, "fc"))
        
        for i, (F, f) in enumerate(zip(self.solver.F, self.solver.f)):
          stg = i / framerate # steps to go
          Fpath = os.path.join(cur_output_folder, "Fm", ("%.5f" % stg).zfill(10))
          np.savetxt(Fpath, F)
          fpath = os.path.join(cur_output_folder, "fc", ("%.5f" % stg).zfill(10))
          np.savetxt(fpath, f)
        
        # write cost function info file
        print ("Writing cost info")
        cost_params = {'terminal_distance': self.terminal_distance,
                       'terminal_velocity': self.terminal_velocity,
                       'control_effort': self.control_effort}
        
        with open(os.path.join(cur_output_folder, "cost_info.yml"), 'w') as outfile:
            outfile.write(yaml.dump(cost_params, default_flow_style=True))

    def analyze(self, output_root):
        tti = self.solver.tti
        F = self.solver.F
        f = self.solver.f
        
        ux = np.zeros((len(F), 2))
        udotx = np.zeros((len(F), 2))
        udotx_agent = np.zeros((len(F), 2))
        uddotx = np.zeros((len(F), 2))
        uddotx_agent = np.zeros((len(F), 2))
        uz = np.zeros((len(F), 2))
        udotz = np.zeros((len(F), 2))
        udotz_agent = np.zeros((len(F), 2))
        uddotz = np.zeros((len(F), 2))
        uddotz_agent = np.zeros((len(F), 2))
        
        u_const = np.zeros((len(f), 3))
        
        yp = np.zeros((len(F), 2))
        yw = np.zeros((len(F), 2))
        
        xdot_symmetric=True
        xddot_symmetric=True
        uconst_symmetric=True
        
        i=0
        for name, M, m in sorted(zip(tti, F, f), key=lambda x: x[0]):
          #print M[0,0]+M[0,6]
          eps = 1e-10
          #if M[0,0] + M[0,6] < eps and M[1,0] + M[1,6] < eps:
          if M[0,0] + M[0,9] < eps and M[1,6] + M[1,12] < eps:
            pass
          else:
            print "%s x and z are NOT symmetric" % name
            return False
            
          if M[0,1] + M[0,10] < eps and M[1,7] + M[1,13] < eps:
            pass
          elif xdot_symmetric:
            print "%s xdot and zdot are NOT symmetric" % name
            xdot_symmetric=False

          # isn't that wrong? M[1,11] is the influence of s_x on u_z!
          # but we want s_z on u_z
          if (M[0,2] + M[0,11]) - (M[1,8] + M[1,14]) < eps:
            pass
          elif xddot_symmetric:
            print "%s xddot and zddot are NOT symmetric" % name
            xddot_symmetric=False
            
          if np.abs(np.abs(m[0]) - np.abs(m[1])) < eps:
            pass
          elif uconst_symmetric:
            print "%s uconst_x and uconst_z are NOT symmetric" % name
            uconst_symmetric=False
            
            
          ux[i, 0] = float(name)
          ux[i, 1] = M[0,0]
          udotx[i, 0] = float(name)
          udotx[i, 1] = M[0,1]
          uddotx[i, 0] = float(name)
          uddotx[i, 1] = M[0,2]
          udotx_agent[i, 0] = float(name)
          udotx_agent[i, 1] = M[0,10]
          uddotx_agent[i, 0] = float(name)
          uddotx_agent[i, 1] = M[0,11]
          
          uz[i, 0] = float(name)
          uz[i, 1] = M[1,6]
          udotz[i, 0] = float(name)
          udotz[i, 1] = M[1,7]
          uddotz[i, 0] = float(name)
          uddotz[i, 1] = M[1,8]
          udotz_agent[i, 0] = float(name)
          udotz_agent[i, 1] = M[1,13]
          uddotz_agent[i, 0] = float(name)
          uddotz_agent[i, 1] = M[1,14]
          
          yp[i,0] = float(name)
          yp[i,1] = M[0,2]
          yw[i,0] = float(name)
          yw[i,1] = M[0,3]
          
          u_const[i,0] = float(name)
          u_const[i,1] = m[0]
          u_const[i,2] = m[1]
          
          i+=1
        
        dt = dynamics.DT
        print r"dt = %f" % dt
        
        #print ux[:,1]
        plt.figure(figsize=(5,3))
        
        #plt.subplot(2,1,1)
        plt.plot(ux[:,0], ux[:,1], lw=3.0, label="$k_p$")
        if xdot_symmetric:
          plt.plot(udotx[:,0], udotx[:,1], lw=3.0, label=r"$k_d$")
        else:
          plt.plot(udotx[:,0], udotx[:,1], lw=3.0, label=r"$k_d \dot{x}_b$")
          plt.plot(udotx_agent[:,0], udotx_agent[:,1], lw=3.0, label=r"$k_d \dot{x}_a$")

        if xddot_symmetric:
          plt.plot(uddotx[:,0], uddotx[:,1], lw=3.0, label=r"$k_a$")
        else:
          plt.plot(uddotx[:,0], uddotx[:,1], lw=3.0, label=r"$k_a \dot{x}_b$")
          plt.plot(uddotx_agent[:,0], uddotx_agent[:,1], lw=3.0, label=r"$k_a \dot{x}_a$")

        # uconst
        #if np.max(u_const[:,1:]) > eps:
        #  plt.plot(u_const[:,0], np.linalg.norm(u_const[:,1:], axis=1), lw=3.0, label="$k_{c}$")
        #else:
        #  print (" uconst is below threshold")            

        #plt.plot(uz[:,0], uz[:,1], lw=2.0, label="uz")
        plt.xlabel(r"\textrm{time to impact}")
        plt.legend()
        #plt.legend(fontsize=24)
        plt.ylim([0, np.max(ux[:,1])*1.1])
        plt.xlim([0, 5.])
        
        fulltitle = r"%s $\Delta t = %.1f$, " % (solver.name, dt)
        cost_term_dct = {\
          'terminal_distance': r"w_\textrm{dist}",
          'terminal_velocity': r"w_\textrm{vel}",
          'control_effort': r"w_\textrm{ctrl}",
        }
        cost_info = {
          'terminal_distance': self.terminal_distance,
          'terminal_velocity': self.terminal_velocity,
          'control_effort': self.control_effort,
        }
        
        if cost_info != None:
          cost_term_dct = dict(reversed(cost_term_dct.items()))
          fulltitle += " " + ", ".join([ "$"+cost_term_dct[k] + "=" + str(v)+"$" for k,v in cost_info.iteritems() if float(v) > 0])

        plt.title(fulltitle)
      
        distfigpath = os.path.join(output_root, "gains_lqr.pdf")
        plt.tight_layout()
        plt.savefig(distfigpath, format='pdf')
        
        np.savetxt(os.path.join(output_root, "_t.txt"), ux[:,0])
        np.savetxt(os.path.join(output_root,"_kp_t.txt"), ux[:,1])
        np.savetxt(os.path.join(output_root,"_kd_t.txt"), udotx[:,1])
        np.savetxt(os.path.join(output_root,"_ka_t.txt"), uddotx[:,1])
        
        # -----------------
        # now also consider time dependence
        # -----------------
       
        plt.figure(figsize=(5,3))
        plt.plot(ux[:,0], ux[:,1]*ux[:,0]*ux[:,0], lw=3.0, ls="-", label=r"$t^2 k_p$")
      
        if xdot_symmetric:
          plt.plot(udotx[:,0], udotx[:,1]*ux[:,0], lw=3.0, ls="-", label=r"$t k_d$")
        else:
          print "NOT SYMMETRIC"
          plt.plot(udotx[:,0], udotx[:,1]*ux[:,0], lw=3.0, ls="-", label=r"$k_d \dot{x}_b$*distance")
          plt.plot(udotx_agent[:,0], udotx_agent[:,1]*ux[:,0], lw=3.0, ls="-", label=r"$k_d \dot{x}_a$*distance")

        # FIXME: ignore acceleration
        #plt.plot(uddotx[:,0], uddotx[:,1]*uddotx[:,0], lw=3.0, label="$t k_{a}$")

        u_norm = u_const[:,1]
        if uconst_symmetric:
          plt.plot(u_const[:,0], u_norm, lw=3.0, label="$k_{c}$")
        else:
          u_norm = np.linalg.norm(u_const[:,1:], axis=1)
          plt.plot(u_const[:,0], u_norm, lw=3.0, label="$\Vert k_{c} \Vert_2$")
      
        #plt.legend(loc="lower right")#, fontsize=24)
        #plt.legend(loc="upper left", ncol=2)#, fontsize=24)
        plt.legend(loc="upper left", ncol=3)#, fontsize=24)
        plt.title(fulltitle)
        plt.xlabel(r"{time to impact (s)}")
        
        xlim = [0,5]
        #mx = np.max([np.max(u_norm[:xlim[1]]), np.max( ux[:xlim[1],1]*ux[:xlim[1],0]*ux[:xlim[1],0])])
        #plt.ylim([0, mx*1.1])
        #plt.ylim([0,3.])
        #plt.ylim([0,10.])
        plt.ylim([0,5.])
        plt.xlim(xlim)
        
        distfigpath = os.path.join(output_root, "gains_t_lqr.pdf")
        print ("Saving "+distfigpath)
        plt.tight_layout()
        plt.savefig(distfigpath, format='pdf')
    

# ====================================================================

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--solver", type=str.lower, choices=["lqr", "ilqr","ilqr_soccer",], default="lqr", help="solver")
      parser.add_argument("--dynamics", type=str.lower, choices=["ideal", "drag",], default="ideal", help="solver")
      parser.add_argument("--framerate", type=float, default=10.,  help="simulation framerate")
      parser.add_argument("--dim", type=int, choices=[2,3], default=3,  help="control dimension")
      parser.add_argument("--terminal_distance", type=float, default=1000.,  help="cost: terminal distance")
      parser.add_argument("--terminal_velocity", type=float, default=0.,  help="cost: terminal velocity")
      parser.add_argument("--control_effort", type=float, default=0.01,  help="cost: control effort")
      parser.add_argument("--no_example", default=False,  action="store_true", help="do not run example")
      parser.add_argument("--no_analysis", default=False,  action="store_true", help="do not analyze the gains")
            
      # TODO expose drag parameters
            
      args = parser.parse_args()

  
      # Init dynamics
      drag = False
      if args.dynamics == "drag":
          drag = True
          
      dynamics = DynamicsModel(framerate=args.framerate, drag=drag, dim=args.dim)
  
      # Init solver
      solver = None
      if args.solver == "lqr":
          solver = LQR()
          dynamics_local = DynamicsModel(framerate=args.framerate, 
                                         drag=False, dim=args.dim, 
                                         copy=True)
      elif "ilqr" in args.solver:
          solver = iLQR()
          if args.solver == "ilqr_soccer":
            dynamics_local = DynamicsModel(framerate=args.framerate, 
                                           #drag=drag, dim=args.dim,
                                           drag=True, dim=args.dim,
                                           mass= 0.4,
                                           c=0.25,
                                           r=0.11,
                                           copy=True)
          else:
            dynamics_local = DynamicsModel(framerate=args.framerate, 
                                           #drag=drag, dim=args.dim,
                                           drag=True, dim=args.dim,
                                           copy=True)

      bc = SOCBallCatching(solver, dynamics_local, \
        args.terminal_distance, args.terminal_velocity, args.control_effort)

      #--------------------------------------
      # RUN SOLVER
      bc.solve()

      #--------------------------------------
      # EXPORT
      output_root = os.path.join(settings_root, args.solver)
      bc.export(output_root)

      #--------------------------------------
      # ANALYZE GAINS

      if not args.no_analysis:
          if not os.path.exists(output_root):
              os.makedirs(output_root)
              
          bc.analyze(output_root)
      
      #--------------------------------------
      # TEST CONTROLLER
      if not args.no_example:
          dt = dynamics.DT  
          g = dynamics.GRAVITY
          D_s = dynamics.state_dim
          D_a = dynamics.action_dim
    
          # start state		
          x0 = np.array([[0.0, 		# 0 -> xb
          			15.556, 	# 1 -> xb' 
                          0.0,    # 2 -> xb''
                          0.0, 	# 3 -> yb
                          15.556,  # 4 -> yb'
                          -g,    # 5 -> yb''
    				0.0, 	# 6 -> zb
    				0.0,  # 7 -> zb'
    				0.0,  # 8 -> zb''
    				63.3374,  # 9 -> xa
    				0.0,  # 10 -> xa'
    				0.0,  # 11 -> xa''
    				0.0,  # 12 -> za
    				0.0,  # 13 -> za'
    				0.0,  # 14 -> za''
    				]]).T
    
          if args.dim == 3:
            x0[12] = 15.556
    
          # run controller
          t, x, u = bc.run(x0)
          
          plt.figure()
          plt.title("position: x of t")
          plt.plot(t, x[:,0], label = "ball x")
          plt.plot(t, x[:,3], label = "ball y")
          plt.plot(t, x[:,9], lw = 2.0 , label = "agent x")
          plt.plot(t, np.zeros(x.shape[0]), c="k", ls="--")
          plt.legend(loc=2)
          
          plt.figure()
          plt.title("position: z of t")
          plt.plot(t, x[:,6], label = "ball z")
          plt.plot(t, x[:,3], label = "ball y")
          plt.plot(t, x[:,12], lw = 2.0 , label = "agent z")
          plt.plot(t, np.zeros(x.shape[0]), c="k", ls="--")
          plt.legend(loc=2)
          
          plt.figure()
          plt.title("velocity: x/z of t")
          plt.plot(t,x[:,3], label = "ball y", c="k")
          plt.plot(t,x[:,10], lw = 2.0 , label = "agent x velocity", c="b")
          plt.plot(t,x[:,13], lw = 2.0 , label = "agent z velocity", c="r")
          plt.plot(t, np.zeros(x.shape[0]), c="k", ls="--")
          plt.legend(loc=2)
          
          plt.figure()
          plt.title("u(t)")
          plt.plot(t,x[:,3], label = "ball y", c="k")
          plt.plot(t[1:],u[:,0], lw = 2.0 , label = "agent x acceleration", c="b")
          plt.plot(t[1:],u[:,1], lw = 2.0 , label = "agent z acceleration", c="r")
          plt.plot(t, np.zeros(x.shape[0]))
          plt.legend(loc=2)
                
          tf = -1
          dist = np.sqrt( (x[tf,0] - x[tf,9])**2 + (x[tf,6] - x[tf,11])**2)
          print "TOTAL DISTANCE, x: %.5f, z: %.5f, total %.5f" % (x[tf,0] - x[tf,9], x[tf,6] - x[tf,12],  dist)
          print ("lqr cost: %.5f" % solver.J(x, u, x.shape[0]))
          
          plt.show()
          