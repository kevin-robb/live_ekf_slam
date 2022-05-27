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
from math import remainder, tau, atan2, pi
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
pp = PurePursuit()
# PID global vars.
integ = 0
err_prev = 0.0
# we're synched to the EKF's "clock", so all time increments are 1 time unit, DT.
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
    # options for navigation functions to use.
    nav_functions = {"pp" : pp_nav,
                     "direct": direct_nav}

    global params
    params_file = open(pkg_path+"/config/params.txt", "r")
    params = {}
    lines = params_file.readlines()
    for line in lines:
        if len(line) < 3 or line[0] == "#": # skip comments and blank lines.
            continue
        p_line = line.split("=")
        key = p_line[0].strip()
        arg = p_line[1].strip()
        # set nav function (string key).
        if key == "NAV_METHOD":
            params["NAV_FUNCTION"] = nav_functions[arg]
            continue
        # set all other params.
        try:
            params[key] = int(arg)
        except:
            try:
                params[key] = float(arg)
            except:
                params[key] = (arg == "True")

    # set coord transform params.
    params["SCALE"] = params["MAP_BOUND"] * 1.5 / (params["OCC_MAP_SIZE"] / 2)
    params["SHIFT"] = params["OCC_MAP_SIZE"] / 2

def norm(l1, l2):
    # compute the norm of the difference of two lists or tuples.
    # only use the first two elements.
    return ((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)**(1/2) 

def pp_nav(state_msg):
    """
    Navigate using pure pursuit.
    """
    global goal_queue, cur, integ, err_prev
    cur = [state_msg.x_v, state_msg.y_v]
    odom_msg = Vector3(x=0, y=0)
    if len(goal_queue) < 1: # if there's no path yet, just wait.
        cmd_pub.publish(odom_msg)
        return

    # define lookahead point.
    lookahead = None
    radius = 0.2 # starting search radius.
    # look until we find the path at the increasing radius or at the maximum dist.
    while lookahead is None and radius <= 3: 
        lookahead = pp.get_lookahead_point(cur[0], cur[1], radius)
        radius *= 1.25
    # make sure we actually found the path.
    if lookahead is not None:
        heading_to_la = atan2(lookahead[1] - cur[1], lookahead[0] - cur[0])
        beta = remainder(heading_to_la - state_msg.yaw_v, tau) # hdg relative to veh pose.

        # update global integral term.
        integ += beta * params["DT"]

        P = 0.08 * beta # proportional to hdg error.
        I = 0.009 * integ # integral to correct systematic error.
        D = 0.01 * (beta - err_prev) / params["DT"] # slope

        # set forward and turning commands.
        odom_msg.x = (1 - abs(beta / pi))**5 + 0.05
        odom_msg.y = P + I + D

        err_prev = beta

    # ensure commands are capped within constraints.
    odom_msg.x = max(0, min(odom_msg.x, params["ODOM_D_MAX"]))
    odom_msg.y = max(-params["ODOM_TH_MAX"], min(odom_msg.y, params["ODOM_TH_MAX"]))
    cmd_pub.publish(odom_msg)


def direct_nav(state_msg):
    """
    Navigate directly point-to-point.
    """
    global goal_queue, cur
    cur = [state_msg.x_v, state_msg.y_v]
    if len(goal_queue) < 1: # if there's no path yet, just wait.
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
    odom_msg = Vector3(x=0, y=0)
    # go faster the more aligned the hdg is.
    odom_msg.x = 1 * (1 - abs(beta)/params["ODOM_TH_MAX"])**3 + 0.05 if r > 0.1 else 0.0
    P = 0.03 if r > 0.2 else 0.2
    odom_msg.y = beta #* P if r > 0.05 else 0.0
    # ensure commands are capped within constraints.
    odom_msg.x = max(0, min(odom_msg.x, params["ODOM_D_MAX"]))
    odom_msg.y = max(-params["ODOM_TH_MAX"], min(odom_msg.y, params["ODOM_TH_MAX"]))
    cmd_pub.publish(odom_msg)
    # remove the goal from the queue if we've arrived.
    if r < 0.15:
        goal_queue.pop(0)
        # publish updated path for the plotter.
        path_pub.publish(Float32MultiArray(data=sum(goal_queue, [])))


def get_ekf_state(msg):
    """
    Get the state published by the EKF.
    Call the desired navigation function.
    """
    params["NAV_FUNCTION"](msg)

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
    if path is not None:
        interpret_astar_path(path)

def cap(ind):
    # cap a map index within 0, max.
    return max(0, min(ind, params["OCC_MAP_SIZE"]-1))

def get_occ_grid_map(msg):
    global occ_map
    # get the true occupancy grid map image.
    bridge = CvBridge()
    occ_map_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # cv2.imshow("Thresholded Map", occ_map); cv2.waitKey(0); cv2.destroyAllWindows()
    # create matrix from map, and flip its x-axis to align with ekf coords better.
    occ_map = []
    for i in range(len(occ_map_img)):
        row = []
        for j in range(len(occ_map_img[0])):
            row.append(int(occ_map_img[i][j]))
        occ_map.append(row)
    # print("raw map:\n",occ_map)
    # determine index pairs to select all neighbors when ballooning obstacles.
    nbrs = []
    for i in range(-params["OCC_MAP_BALLOON_AMT"], params["OCC_MAP_BALLOON_AMT"]+1):
        for j in range(-params["OCC_MAP_BALLOON_AMT"], params["OCC_MAP_BALLOON_AMT"]+1):
            nbrs.append((i, j))
    # remove 0,0 which is just the parent cell.
    nbrs.remove((0,0))
    # expand all occluded cells outwards.
    for i in range(len(occ_map)):
        for j in range(len(occ_map[0])):
            if occ_map_img[i][j] < 0.5: # occluded.
                # mark all neighbors as occluded.
                for chg in nbrs:
                    occ_map[cap(i+chg[0])][cap(j+chg[1])] = 0
    # print("inflated map:\n",occ_map)
    # show value distribution in occ_map.
    freqs = [0, 0]
    for i in range(len(occ_map)):
        for j in range(len(occ_map[0])):
            if occ_map[i][j] == 0:
                freqs[0] += 1
            else:
                freqs[1] += 1
    print(CYAN+"Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded."+ANSI_RESET)

def tf_map_to_ekf(pt):
    # transform x,y from occ map indices to ekf coords.
    return [(pt[1] - params["SHIFT"]) * params["SCALE"], -(pt[0] - params["SHIFT"]) * params["SCALE"]]
    
def tf_ekf_to_map(pt):
    # transform x,y from ekf coords to occ map indices.
    return [int(params["SHIFT"] - pt[1] / params["SCALE"]), int(params["SHIFT"] + pt[0] / params["SCALE"])]

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
        goal_queue.append(tf_map_to_ekf(path_to_start[i]))
        pp.add_point(tf_map_to_ekf(path_to_start[i]))
    # publish this path for the plotter.
    path_pub.publish(Float32MultiArray(data=sum(goal_queue, [])))

def astar(goal):
    """
    Use A* to generate a path from the current pose to the goal position.
    1 map grid cell = 0.1x0.1 units in ekf coords.
    map (0,0) = ekf (-10,10).
    """
    # define goal node.
    goal_cell = Cell(tf_ekf_to_map(goal))
    # reject the goal pt if it's in an occluded cell.
    if occ_map[goal_cell.i][goal_cell.j] == 0:
        rospy.logerr("Invalid goal point (in collision).")
        return
    # first add starting node to open list.
    if len(goal_queue) > 0: # use end of prev segment as start if there is one.
        start_cell = Cell(tf_ekf_to_map(goal_queue[-1]))
    else: # otherwise use the current position estimate.
        start_cell = Cell(tf_ekf_to_map(cur))
    open_list = [start_cell]
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
            nbr = Cell([cur_cell.i+chg[0], cur_cell.j+chg[1]], parent=cur_cell)
            # skip if out of bounds.
            if nbr.i < 0 or nbr.j < 0 or nbr.i >= params["OCC_MAP_SIZE"] or nbr.j >= params["OCC_MAP_SIZE"]: continue
            # skip if occluded.
            if occ_map[nbr.i][nbr.j] == 0: continue
            # skip if already in closed list.
            if any([nbr == c for c in closed_list]): continue
            # skip if already in open list, unless the cost is lower.
            seen = [nbr == open_cell for open_cell in open_list]
            try:
                match_i = seen.index(True)
                # this cell has been added to the open list already.
                # check if the new or existing route here is better.
                if nbr.g < open_list[match_i].g: 
                    # the cell has a shorter path this new way, so update its cost and parent.
                    open_list[match_i].set_cost(g=nbr.g)
                    open_list[match_i].parent = nbr.parent
                continue
            except:
                # there's no match, so proceed.
                pass
            # compute heuristic "cost-to-go". keep squared to save unnecessary computation.
            nbr.set_cost(h=(goal_cell.i - nbr.i)**2 + (goal_cell.j - nbr.j)**2)
            # add cell to open list.
            open_list.append(nbr)

class Cell:
    def __init__(self, pos, parent=None):
        self.i = int(pos[0])
        self.j = int(pos[1])
        self.parent = parent
        self.g = 0 if parent is None else parent.g + 1
        self.f = 0
    
    def set_cost(self, h=None, g=None):
        # set/update either g or h and recompute the cost, f.
        if h is not None:
            self.h = h
        if g is not None:
            self.g = g
        # update the cost.
        self.f = self.g + self.h

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