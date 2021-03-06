#!/usr/bin/env python3

"""
Particle Filter implementation
using Monte Carlo Localization.
"""

import numpy as np
from random import random, choices
from math import sin, cos, remainder, tau, atan2, pi, log, exp
from scipy.stats import multivariate_normal, norm, uniform
from base_pkg.msg import PFState

class PF:
    is_init = False

    def __init__(self, params):
        self.timestep = 0
        self.params = params
        # create process and sensing noise covariance matrices.
        self.V = np.array([[params["V_00"],0.0],[0.0,params["V_11"]]])
        self.W = np.array([[params["W_00"],0.0],[0.0,params["W_11"]]])

    def init_particle_set(self, x_init):
        """
        Initialize the particles uniformly across the entire map.
        """
        self.x_t = np.empty((3,self.params["NUM_PARTICLES"]))
        if self.params["USE_INIT_VEH_POSE"]:
            # create all particles clustered around known veh starting pose.
            for i in range(self.params["NUM_PARTICLES"]):
                self.x_t[0,i] = x_init[0] + (0.1 * random() - 0.05)
                self.x_t[1,i] = x_init[1] + (0.1 * random() - 0.05)
                self.x_t[2,i] = x_init[2] + (0.1 * random() - 0.05)
        else:
            # create particles spread randomly across the map.
            for i in range(self.params["NUM_PARTICLES"]):
                self.x_t[0,i] = 2*self.params["MAP_BOUND"]*random() - self.params["MAP_BOUND"]
                self.x_t[1,i] = 2*self.params["MAP_BOUND"]*random() - self.params["MAP_BOUND"]
                self.x_t[2,i] = 2*pi*random() - pi
        # instantiate predictions set as well.
        self.x_pred = self.x_t
        self.is_init = True


    def iterate(self, u, z):
        """
        Perform one iteration of MCL.
        """
        # update timestep.
        self.timestep += 1
        # propagate particles forward one timestep.
        self.sample_motion_model(u)
        # skip update step if no landmarks were detected.
        if len(z) < 1:
            self.x_t = self.x_pred
            return
        # compute particle log-likelihood weights.
        wts = [self.sensor_likelihood(z, i) for i in range(self.params["NUM_PARTICLES"])]
        # compute LSE term.
        wt_mean = sum(wts) / len(wts)
        lse = log(sum([exp(wts[i]-wt_mean) for i in range(self.params["NUM_PARTICLES"])]))
        # compute probabilities.
        prbs = [exp(wt - wt_mean - lse) for wt in wts]

        # cull particles that have likely left the map.
        for i in range(self.params["NUM_PARTICLES"]):
            if max(abs(self.x_pred[0,i]), abs(self.x_pred[1,i])) > (1.5 * self.params["MAP_BOUND"]):
                prbs[i] = 0
        # normalize probabilities.
        prb_tot = sum(prbs)
        prbs = [prbs[i] / prb_tot for i in range(self.params["NUM_PARTICLES"])]
        # resample particles based on weights.
        ind_set = choices(list(range(self.params["NUM_PARTICLES"])), weights=prbs, k=self.params["NUM_PARTICLES"])
        self.x_t = np.vstack(tuple([self.x_pred[0:3,i] for i in ind_set])).T
        # randomly replace some particles with new, random ones.
        ind_to_replace = choices(list(range(self.params["NUM_PARTICLES"])), k=self.params["NUM_PARTICLES"]//100)
        for i in ind_to_replace:
            self.x_t[0,i] = 2*self.params["MAP_BOUND"]*random() - self.params["MAP_BOUND"]
            self.x_t[1,i] = 2*self.params["MAP_BOUND"]*random() - self.params["MAP_BOUND"]
            self.x_t[2,i] = 2*pi*random() - pi


    def sample_motion_model(self, u):
        """
        Implementation of motion model sampling function.
        Perform forward kinematics for one timestep for
        all particles using this command, u.
        """
        # ensure we start with current set.
        self.x_pred = self.x_t
        # extract odometry commands.
        d_d = u[0]; d_th = u[1]
        # create process noise distribution.
        noise = multivariate_normal.rvs(mean=None, cov=self.V, size=self.params["NUM_PARTICLES"])
        for i in range(self.params["NUM_PARTICLES"]):
            # sample a random realization of the process noise. 
            # apply motion model to predict next pose for this particle.
            self.x_pred[0,i] = self.x_t[0,i] + (d_d + noise[i][0])*cos(self.x_t[2,i])
            self.x_pred[1,i] = self.x_t[1,i] + (d_d + noise[i][0])*sin(self.x_t[2,i])
            # cap heading to (-pi,pi).
            self.x_pred[2,0] = remainder(self.x_t[2,0]+d_th+noise[i][1], tau)


    def sensor_likelihood(self, z, i:int):
        """
        Given the true landmark measurements z,
        find the log likelihood of measuring this scan if 
        particle i is correct. Use the known map.
        """
        # in a real scenario there's always a nonzero chance.
        wt_rand = 0.05
        # for each observed landmark, compute the distance we'd expect given the map and particle i's pose.
        log_likelihood = 0
        for l in range(len(z) // 3):
            # likelihood should be ~ zero if range is further than max vision range or bearing is outside fov.
            r_exp, b_exp = self.raycast(z[3*l], i)
            r_distr = (1-wt_rand)*norm.cdf(z[3*l+1], loc=r_exp, scale=self.W[0,0]) + wt_rand*uniform.cdf(z[3*l+1], loc=0, scale=self.params["RANGE_MAX"])
            b_distr = (1-wt_rand)*norm.cdf(z[3*l+2], loc=b_exp, scale=self.W[1,1]) + wt_rand*uniform.cdf(z[3*l+2], loc=self.params["FOV_MIN"], scale=self.params["FOV_MAX"]-self.params["FOV_MIN"])
            log_likelihood += log(r_distr) + log(b_distr)

        return log_likelihood


    def raycast(self, l:int, i:int):
        """
        Compute the expected measurements r, b
        from particle i to landmark l.
        """
        r = ((self.MAP[l][0]-self.x_pred[0,i])**2 + (self.MAP[l][1]-self.x_pred[1,i])**2)**(1/2)
        b = remainder(atan2(self.MAP[l][1]-self.x_pred[1,i], self.MAP[l][0]-self.x_pred[0,i])-self.x_pred[2,i], tau)
        return r, b


    def get_state(self):
        """
        Return the current set of particles as a PFState msg.
        """
        msg = PFState()
        msg.timestep = self.timestep
        msg.x = [self.x_t[0,i] for i in range(self.params["NUM_PARTICLES"])]
        msg.y = [self.x_t[1,i] for i in range(self.params["NUM_PARTICLES"])]
        msg.yaw = [self.x_t[2,i] for i in range(self.params["NUM_PARTICLES"])]
        # TODO look into publishing weights as well.
        return msg