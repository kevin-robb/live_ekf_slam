#!/usr/bin/env python3

"""
Command vehicle to pursue path to goal point chosen by clicking the plot.
Generate the next odom command based on current state estimate.
"""

import rospy
import rospkg
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from ekf_pkg.msg import EKFState
from pf_pkg.msg import PFState
from math import remainder, tau, atan2
import cv2
from cv_bridge import CvBridge
from pure_pursuit import PurePursuit

############ GLOBAL VARIABLES ###################
params = {}
cmd_pub = None; path_pub = None
# goal = [0.0, 0.0] # target x,y point.
goal_queue = [] # queue of goal points to force a certain path.
cur = [0.0, 0.0] # current estimate of veh pos.
occ_map = None # cv2 image of true occupancy grid map.
# init pure pursuit object.
# pp = PurePursuit()
#################################################
# Color codes for console output.
ANSI_RESET = "\u001B[0m"
GREEN = "\u001B[32m"
YELLOW = "\u001B[33m"
BLUE = "\u001B[34m"
PURPLE = "\u001B[35m"
CYAN = "\u001B[36m"
#################################################


def read_params(pkg_path):
    """
    Read params from config file.
    @param path to data_pkg.
    """
    global params, shift, scale
    params_file = open(pkg_path+"/config/params.txt", "r")
    params = {}
    lines = params_file.readlines()
    for line in lines:
        if len(line) < 3 or line[0] == "#": # skip comments and blank lines.
            continue
        p_line = line.split("=")
        key = p_line[0].strip()
        arg = p_line[1].strip()
        try:
            params[key] = int(arg)
        except:
            try:
                params[key] = float(arg)
            except:
                params[key] = (arg == "True")

    # set coord transform params.
    scale = params["MAP_BOUND"] * 1.5 / (params["OCC_MAP_SIZE"] / 2)
    shift = params["OCC_MAP_SIZE"] / 2

def norm(l1, l2):
    # compute the norm of the difference of two lists or tuples.
    # only use the first two elements.
    return ((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)**(1/2) 

def choose_command(state_msg):
    """
    Given the current state estimate and a goal position,
    choose and send an odom cmd to the vehicle.
    """
    global goal_queue, cur
    cur = [state_msg.x_v, state_msg.y_v]
    if len(goal_queue) < 1:
        cmd_pub.publish(Vector3(x=0, y=0))
        return
    goal = goal_queue[0]
    # check how close veh is to our current goal pt.
    r = norm([state_msg.x_v, state_msg.y_v], goal)
    # compute vector from veh pos est to goal.
    diff_vec = [goal[0]-state_msg.x_v, goal[1]-state_msg.y_v]
    # calculate bearing difference.
    gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
    beta = remainder(gb - state_msg.yaw_v, tau) # bearing rel to robot
    # turn this into a command.
    odom_msg = Vector3()
    # go faster the more aligned the hdg is.
    odom_msg.x = 0.1 * (1 - abs(beta)/params["ODOM_TH_MAX"])**5 + 0.05 if r > 0.1 else 0.0
    P = 0.03 if r > 0.2 else 0.2
    odom_msg.y = beta * P if r > 0.05 else 0.0
    # # use pure pursuit to get heading.
    # hdg = pp_nav()
    # odom_msg.y = hdg
    # ensure commands are capped within constraints.
    odom_msg.x = max(0, min(odom_msg.x, params["ODOM_D_MAX"]))
    odom_msg.y = max(-params["ODOM_TH_MAX"], min(odom_msg.y, params["ODOM_TH_MAX"]))
    cmd_pub.publish(odom_msg)
    # remove the goal from the queue if we've arrived.
    if r < 0.15:
        goal_queue.pop(0)
        # rospy.logwarn("Arrived at current goal!")
        # publish updated path for the plotter.
        path_pub.publish(Float32MultiArray(data=sum(goal_queue, [])))


def get_ekf_state(msg):
    """
    get the state published by the EKF.
    """
    choose_command(msg)

def get_pf_state(msg):
    """
    get the state published by the PF.
    """
    # TODO
    pass
    
def get_goal_pt(msg):
    # global goal
    goal = [msg.x, msg.y]
    rospy.loginfo("Setting goal pt to ("+"{0:.4f}".format(msg.x)+", "+"{0:.4f}".format(msg.y)+")")
    # generate path there with A*.
    path = astar(goal)
    interpret_astar_path(path)

def get_occ_grid_map(msg):
    global occ_map
    # get the true occupancy grid map image.
    bridge = CvBridge()
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # cv2.imshow("Thresholded Map", occ_map); cv2.waitKey(0); cv2.destroyAllWindows()

def tf_map_to_ekf(component):
    # transform x or y component from  occ map indices to ekf coords.
    return (component - shift) * scale
    
def tf_ekf_to_map(component):
    # transform x or y component from ekf coords to occ map indices.
    return component / scale + shift

# def pp_nav():
#     """
#     Navigate using pure pursuit.
#     """
#     lookahead = None
#     # start with this search radius.
#     radius = 0.4
#     # look until we find the path at the increasing radius or at the maximum dist.
#     while lookahead is None and radius <= 6: 
#         lookahead = pp.get_lookahead_point(cur[0], cur[1], radius)
#         radius *= 1.25
#     # make sure we actually found the path
#     if lookahead is not None:
#         heading_to_la = remainder(atan2(lookahead[1] - cur[1], lookahead[0] - cur[0]), tau)

#         return heading_to_la
#     return None


def interpret_astar_path(path_to_start):
    """
    A* gives a reverse list of index positions to get to goal.
    Reverse this, convert to EKF coords, and add all to goal_queue.
    """
    global goal_queue
    if path_to_start is None:
        rospy.logerr("No path found by A*.")
        return
    for i in range(len(path_to_start)-1, -1, -1):
        # convert pt to ekf coords.
        goal_queue.append([tf_map_to_ekf(path_to_start[i][0]), tf_map_to_ekf(path_to_start[i][1])])
        # pp.add_point(tf_map_to_ekf(path_to_start[i][0]), tf_map_to_ekf(path_to_start[i][1]))
    print("Found goal path: "+str(goal_queue))
    # publish this path for the plotter.
    path_pub.publish(Float32MultiArray(data=sum(goal_queue, [])))

def astar(goal):
    """
    Use A* to generate a path from the current pose to the goal position.
    1 map grid cell = 0.1x0.1 units in ekf coords.
    map (0,0) = ekf (-10,10).
    """
    # define goal node.
    goal_cell = Cell(tf_ekf_to_map(goal[0]), tf_ekf_to_map(goal[1]))
    print("Goal:"+str(goal_cell))
    rospy.logwarn("Map val at goal: "+str(occ_map[goal_cell.i][goal_cell.j]))
    # first add starting node to open list.
    if len(goal_queue) > 0: # use end of prev segment as start if there is one.
        start_cell = Cell(tf_ekf_to_map(goal_queue[-1][0]), tf_ekf_to_map(goal_queue[-1][1]))
    else: # otherwise use the current position estimate.
        start_cell = Cell(tf_ekf_to_map(cur[0]), tf_ekf_to_map(cur[1]))
    open_list = [start_cell]
    print("Start:"+str(open_list[0]))
    closed_list = []
    # iterate until reaching the goal or exhausting all cells.
    while len(open_list) > 0:
        # move first element of open list to closed list.
        open_list.sort(key=lambda cell: cell.f)
        cur_cell = open_list.pop(0)
        # stop if we've found the goal.
        if cur_cell == goal_cell:
            # recurse up thru parents to get reverse of path from start to goal.
            path_to_start = []
            while cur_cell.parent is not None:
                path_to_start.append((cur_cell.i, cur_cell.j))
                cur_cell = cur_cell.parent
            return path_to_start
        # add this node to the closed list.
        closed_list.append(cur_cell)
        # add its unoccupied neighbors to the open list.
        for chg in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            # check each cell for occlusion and if we've already checked it.
            nbr = Cell(cur_cell.i+chg[0], cur_cell.j+chg[1], parent=cur_cell)
            # skip if out of bounds.
            if nbr.i < 0 or nbr.j < 0 or nbr.i >= params["OCC_MAP_SIZE"] or nbr.j >= params["OCC_MAP_SIZE"]: continue
            # skip if occluded.
            if occ_map[nbr.i][nbr.j] != 1:
                # rospy.logwarn("Neighbor is occluded.")
                continue
            # skip if already in closed list.
            if any([nbr == c for c in closed_list]): continue
            # skip if already in open list, unless the cost is lower.
            for open_cell in open_list:
                if nbr == open_cell:
                    if nbr.g > open_cell.g:
                        continue
                    break
            # compute heuristic "cost-to-go". keep squared to save unnecessary computation.
            nbr.set_cost((goal_cell.i - nbr.i)**2 + (goal_cell.j - nbr.j)**2)
            # add cell to open list.
            print("Added open cell: "+str(nbr))
            open_list.append(nbr)

class Cell:
    def __init__(self, i, j, parent=None):
        self.i = int(i)
        self.j = int(j)
        self.parent = parent
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0
    
    def set_cost(self, h):
        self.h = h
        self.f = self.g + h

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def __str__(self):
        return "Cell ("+str(self.i)+","+str(self.j)+") with costs "+str([self.g, self.f])


def main():
    global cmd_pub, path_pub
    rospy.init_node('goal_pursuit_node')

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')
    # read params.
    read_params(pkg_path)

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    rospy.Subscriber("/state/pf", PFState, get_pf_state, queue_size=1)
    # subscribe to the true occupancy grid.
    rospy.Subscriber("/truth/occ_grid", Image, get_occ_grid_map, queue_size=1)

    # publish odom commands for the vehicle.
    cmd_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

    # subscribe to current goal point.
    rospy.Subscriber("/plan/goal", Vector3, get_goal_pt, queue_size=1)
    # publish planned path to the goal.
    path_pub = rospy.Publisher("/plan/path", Float32MultiArray, queue_size=1)


    # instruct the user.
    rospy.loginfo("Left-click on the plot to set the vehicle's goal position.")

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass