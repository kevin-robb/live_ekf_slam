#!/usr/bin/env python3

"""
Particle Filter implementation
using Monte Carlo Localization.
"""

import numpy as np
from random import random, choices
from math import sin, cos, remainder, tau, atan2, pi
from scipy.stats import multivariate_normal


class PF:
    ##### CONFIG VALUES #####
    # Process noise in (forward, angular). Assume zero mean.
    v_d = 0; v_th = 0
    # process noise covariance: symmetric positive-definite matrix.
    V = np.array([[0.02**2,0.0],[0.0,(0.5*pi/180)**2]])
    # Sensing noise in (range, bearing). Assume zero mean.
    w_r = 0; w_b = 0
    W = np.array([[0.1**2,0.0],[0.0,(1*pi/180)**2]])


    def __init__(self, num_particles, DT, map, map_bounds = (-10,10)):
        """
        Initialize the particles uniformly across the entire map.map_bounds
        True map is available
            -> Randomly select locations in the world.
            -> Assign particles to those locations along with a random orientation value in (-pi, pi).
        """
        self.NUM_PARTICLES = num_particles
        self.DT = DT
        self.MAP = map
        self.MAP_BOUNDS = map_bounds
        # create particle set, x_t.
        self.x_t = np.empty((3,num_particles))
        for i in range(num_particles):
            self.x_t[0,i] = (map_bounds[1]-map_bounds[0])*random() + map_bounds[0]
            self.x_t[1,i] = (map_bounds[1]-map_bounds[0])*random() + map_bounds[0]
            self.x_t[1,i] = 2*pi*random() - pi
        # instantiate predictions set as well.
        self.x_pred = self.x_t


    def iterate(self, u, z):
        """
        Perform one iteration of MCL.
        """
        # propagate particles forward one timestep.
        self.sample_motion_model(u)
        # compute particle weights and normalize them.
        wts = [self.sensor_likelihood(z, i) for i in range(self.NUM_PARTICLES)]
        # normalize weights.
        wt_tot = sum(wts)
        wts = [wts[w] / wt_tot for w in range(self.NUM_PARTICLES)]
        # resample particles based on weights.
        ind_set = choices(list(range(self.NUM_PARTICLES)), weights=wts, k=self.NUM_PARTICLES)
        self.x_t = np.vstack(tuple([self.x_pred[0:3,i] for i in ind_set])).T


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
        noise = multivariate_normal.rvs(mean=None, cov=self.V, size=self.NUM_PARTICLES)
        for i in range(self.NUM_PARTICLES):
            # sample a random realization of the process noise. 
            # apply motion model to predict next pose for this particle.
            self.x_pred[0,i] = self.x_t[0,i] + (d_d + noise[i][0])*cos(self.x_t[2,i])
            self.x_pred[1,i] = self.x_t[1,i] + (d_d + noise[i][0])*sin(self.x_t[2,i])
            # cap heading to (-pi,pi).
            self.x_pred[2,0] = remainder(self.x_t[2,0]+d_th+noise[i][1], tau)


    def sensor_likelihood(self, z, i:int):
        """
        Given the true landmark measurements z,
        find the likelihood of measuring this scan if particle i
        is correct. Use the known map.
        """
        # for each observed landmark, compute the distance we'd expect given the map and particle i's pose.
        error = 0
        for l in range(len(z) // 3):
            r_exp, b_exp = self.raycast(l, i)
            r_err = abs(z[l+1] - r_exp)
            b_err = abs(z[l+2] - b_exp)
            error += r_err + b_err
        return 1 / error


    def raycast(self, l:int, i:int):
        """
        Compute the expected measurements r, b
        from particle i to landmark l.
        """
        r = ((self.MAP[l][0]-self.x_pred[0,i])**2 + (self.MAP[l][1]-self.x_pred[1,i])**2)**(1/2)
        b = remainder(atan2(self.MAP[l][1]-self.x_pred[1,i], self.MAP[l][0]-self.x_pred[0,i])-self.x_pred[2,i], tau)
        return r, b

    def get_particle_set(self):
        """
        Return the current set of particles as a 1x3N list.
        """
        pset = []
        for i in range(self.NUM_PARTICLES):
            pset += [self.x_t[0,i], self.x_t[1,i], self.x_t[2,i]]
        return pset