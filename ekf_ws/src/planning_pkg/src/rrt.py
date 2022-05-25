#!/usr/bin/env python3

from random import random
from math import remainder, atan2, tau, sin, cos, floor

"""
Use RRT taking motion constraints into account to find a path to goal.
"""

class RRT:
    # list of nodes. Each node stores ID for node it came from.
    tree = []
    x_v = [0,0,0]
    def __init__(self, x_v, y_v, yaw_v, params, map):
        # starting node is robot's current pose.
        self.tree = [Node(x_v, y_v, yaw_v, 0)]
        # will need to ensure the path is w/in constraints and has no collisions.
        self.params = params
        self.map = map

    def check_collision(self):
        # check if a path is in collision with the map.
        return False

    def find_path(self, x_g, y_g):
        # generate RRT until a path to the goal is found.
        # x_v = -10; y_v = -10; yaw_v = 0.0
        while True:
            # randomly choose point to strive towards. sometimes use final goal.
            goal = (20*random()-10, 20*random()-10) if random() > 0.1 else (x_g, y_g)
            # randomly choose a node to start from. TODO should choose closest pt to goal.
            node_id = floor(len(self.tree)*random())
            x_v = self.tree[node_id].x
            y_v = self.tree[node_id].y
            yaw_v = self.tree[node_id].yaw
            # determine angle to this pt.
            diff_vec = (goal[0]-x_v, goal[1]-y_v)
            gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
            beta = remainder(gb - self.x_v[2], tau) # bearing rel to robot
            # cap to within constraints.
            beta = max(-self.params["ODOM_TH_MAX"], max(beta, self.params["ODOM_TH_MAX"]))
            # determine distance to pt.
            range = (diff_vec[0]**2 + diff_vec[1]**2)**(1/2)
            # cap within constraints.
            range = max(-self.params["ODOM_D_MAX"], max(range, self.params["ODOM_D_MAX"]))
            # if this motion is not in collision, add a node.
            if not self.check_collision():
                self.tree.append(Node(x_v+range*cos(beta), y_v+range*sin(beta), yaw_v+beta, node_id))

            return None


class Node:
    # Node in the RRT.
    # stores its own state + node it came from.
    def __init__(self, x:float, y:float, yaw:float, parent):
        self.x = x
        self.y = y
        self.yaw = yaw # radians (-pi,pi), 0 = facing right.
        self.parent_ID = parent
        self.children = []

    def add_child(self, child):
        # add another node that follows this one.
        self.children.append(child)