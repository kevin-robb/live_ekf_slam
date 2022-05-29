#!/usr/bin/env python3

"""
Command vehicle to pursue path to goal point chosen by clicking the plot.
Generate the next odom command based on current state estimate.
"""

import rospy
import rospkg
from data_pkg.msg import Command
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from ekf_pkg.msg import EKFState
from pf_pkg.msg import PFState
from cv_bridge import CvBridge
from pure_pursuit import PurePursuit
from astar import Astar
from import_params import read_params

############ GLOBAL VARIABLES ###################
params = {}
cmd_pub = None; path_pub = None
# goal = [0.0, 0.0] # target x,y point.
cur = [0.0, 0.0] # current estimate of veh pos.
occ_map = None # cv2 image of true occupancy grid map.
#################################################
# Color codes for console output.
ANSI_RESET = "\u001B[0m"
ANSI_GREEN = "\u001B[32m"
ANSI_YELLOW = "\u001B[33m"
ANSI_BLUE = "\u001B[34m"
ANSI_PURPLE = "\u001B[35m"
ANSI_CYAN = "\u001B[36m"
#################################################


# def read_params(pkg_path):
#     """
#     Read params from config file.
#     @param path to data_pkg.
#     """
#     global params
#     params_file = open(pkg_path+"/config/params.txt", "r")
#     params = {}
#     lines = params_file.readlines()
#     for line in lines:
#         if len(line) < 3 or line[0] == "#": # skip comments and blank lines.
#             continue
#         p_line = line.split("=")
#         key = p_line[0].strip()
#         arg = p_line[1].strip()
#         # set nav function (string key).
#         if key == "NAV_METHOD":
#             params["NAV_METHOD"] = arg
#             continue
#         # set all other params.
#         try:
#             params[key] = int(arg)
#         except:
#             try:
#                 params[key] = float(arg)
#             except:
#                 params[key] = (arg == "True")

#     # set coord transform params.
#     params["SCALE"] = params["MAP_BOUND"] * 1.5 / (params["OCC_MAP_SIZE"] / 2)
#     params["SHIFT"] = params["OCC_MAP_SIZE"] / 2

#     # set params for other functions to access too.
#     Astar.params = params
#     PurePursuit.params = params


def get_ekf_state(msg):
    """
    Get the state published by the EKF.
    """
    # update current position.
    global cur
    cur = [msg.x_v, msg.y_v, msg.yaw_v]
    # check size of path.
    path_len = len(PurePursuit.goal_queue)
    # call the desired navigation function.
    if params["NAV_METHOD"] == "pp":
        # use pure pursuit.
        cmd_pub.publish(PurePursuit.get_next_cmd(cur))
    elif params["NAV_METHOD"] in ["direct", "simple"]:
        # directly go to each point.
        cmd_pub.publish(PurePursuit.direct_nav(cur))
    else:
        rospy.logerr("Invalid NAV_METHOD choice.")
        exit()
    # if the path length has changed (by a point being reached), update the plot.
    if len(PurePursuit.goal_queue) != path_len:
        path_pub.publish(Float32MultiArray(data=sum(PurePursuit.goal_queue, [])))


def get_goal_pt(msg):
    # verify chosen pt is not in collision.
    if occ_map[Astar.tf_ekf_to_map((msg.x,msg.y))[0]][Astar.tf_ekf_to_map((msg.x,msg.y))[1]] == 0:
        rospy.logerr("Invalid goal point (in collision).")
        return
    rospy.loginfo("Setting goal pt to ("+"{0:.4f}".format(msg.x)+", "+"{0:.4f}".format(msg.y)+")")
    # if running in "simple" mode, only add goal point rather than path planning.
    if params["NAV_METHOD"] == "simple":
        PurePursuit.goal_queue.append([msg.x, msg.y])
        return
    # determine starting pos for path.
    if len(PurePursuit.goal_queue) > 0: # use end of prev segment as start if there is one.
        start = PurePursuit.goal_queue[-1]
    else: # otherwise use the current position estimate.
        start = cur
    # generate path with A*.
    path = Astar.astar(start, [msg.x, msg.y])
    if path is None:
        rospy.logerr("No path found by A*.")
        return
    # turn this path into a list of positions for goal_queue.
    new_path_segment = Astar.interpret_astar_path(path)
    # add these to the pure pursuit path.
    PurePursuit.goal_queue += new_path_segment
    # publish this path for the plotter to display.
    path_pub.publish(Float32MultiArray(data=sum(PurePursuit.goal_queue, [])))


def get_occ_grid_map(msg):
    global occ_map
    # get the true occupancy grid map image.
    bridge = CvBridge()
    occ_map_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
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
                    occ_map[max(0, min(i+chg[0], params["OCC_MAP_SIZE"]-1))][max(0, min(j+chg[1], params["OCC_MAP_SIZE"]-1))] = 0
    # print("inflated map:\n",occ_map)
    # show value distribution in occ_map.
    freqs = [0, 0]
    for i in range(len(occ_map)):
        for j in range(len(occ_map[0])):
            if occ_map[i][j] == 0:
                freqs[0] += 1
            else:
                freqs[1] += 1
    print(ANSI_CYAN+"Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded."+ANSI_RESET)
    # set map for A* to use.
    Astar.occ_map = occ_map


def main():
    global cmd_pub, path_pub, params
    rospy.init_node('goal_pursuit_node')

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')
    # read params.
    params = read_params()
    # set coord transform params.
    params["SCALE"] = params["MAP_BOUND"] * 1.5 / (params["OCC_MAP_SIZE"] / 2)
    params["SHIFT"] = params["OCC_MAP_SIZE"] / 2
    # set params for other functions to access too.
    Astar.params = params
    PurePursuit.params = params

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    # rospy.Subscriber("/state/pf", PFState, get_pf_state, queue_size=1)
    # subscribe to the true occupancy grid.
    rospy.Subscriber("/truth/occ_grid", Image, get_occ_grid_map, queue_size=1)

    # publish odom commands for the vehicle.
    cmd_pub = rospy.Publisher("/command", Command, queue_size=1)

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