#!/usr/bin/env python3

"""
Framework for Pose-Graph SLAM
"""

from math import cos, sin, remainder, tau, pi
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

class PGSlam:
    pose_graph = {}
    constraints = []
    # latest position est to use for next motion start.
    x_t = None

    def __init__(self, params, x_0,y_0,yaw_0):
        self.params = params
        self.x_t = [x_0,y_0,yaw_0]
        # add initial position constraint.
        # self.constraints.append()

    def iterate(self, cmd, lm_meas):
        """
        Accept a new motion cmd + lm meas.
        Create new node for current pos.
        Check for loop closure and adjust pose graph as needed.
        """
        u_d = cmd[0]; u_th = cmd[1]
        # motion model.
        dx = (u_d + self.params["v_d"])*cos(self.x_t[2])
        dy = (u_d + self.params["v_d"])*sin(self.x_t[2])
        dyaw = u_th + self.params["v_th"]
        # create relative motion constraint.


    def identify_loop_closure(self):
        """
        Determine whether the most recent pose is observing a previously-seen
        part of the map, and is eligible for a loop closure.
        """
        # TODO
        return False

    def rel_motion_constraints_fun(self):
        """
        Function to compute relative motion constraints for optimization.
        """
        

class Node:
    """
    A single pose + measurement, constituting a node in the pose-graph.
    """
    def __init__(self, timestep, x,y,yaw, id,r,b):
        # timestep serves as identifier for the node.
        self.id = timestep
        self.pose = [x,y,yaw]
        self.meas = [id,r,b]

class Connection:
    """
    A Connection between two nodes.
    """
    def __init__(self, dx,dy,dth):
        pass