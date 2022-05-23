#!/usr/bin/env python3

"""
Node to track true state and send landmark measurements
to the EKF node, as well as LiDAR measurements to
the mapping node for occupancy grid creation.
Receives odom commands and performs them with noise.
"""

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
import sys
from random import random
from math import atan2, remainder, tau, cos, sin
import numpy as np

############ GLOBAL VARIABLES ###################
params = {}
# publishers
lm_pub = None; true_map_pub = None; true_pose_pub = None; cmd_pub = None
# Default map from RSS demo:
demo_map = { 0 : (6.2945, 8.1158), 1 : (-7.4603, 8.2675), 2 : (2.6472, -8.0492), 
        3 : (-4.4300, 0.9376), 4 : (9.1501, 9.2978), 5 : (-6.8477, 9.4119), 6 : (9.1433, -0.2925),
        7 : (6.0056, -7.1623), 8 : (-1.5648, 8.3147), 9 : (5.8441, 9.1898), 10: (3.1148, -9.2858),
        11: (6.9826, 8.6799), 12: (3.5747, 5.1548), 13: (4.8626, -2.1555), 14: (3.1096, -6.5763),
        15: (4.1209, -9.3633), 16: (-4.4615, -9.0766), 17: (-8.0574, 6.4692), 18: (3.8966, -3.6580), 19: (9.0044, -9.3111) }
# true map and current pose.
landmarks = None; x_v = [0.0, 0.0, 0.0]
#################################################


def read_params(pkg_path):
    """
    Read params from config file.
    @param path to data_pkg.
    """
    global params, x_v
    params_file = open(pkg_path+"/config/params.txt", "r")
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
    # init vehicle pose.
    x_v = [params["x_0"], params["y_0"], params["yaw_0"]]


def norm(l1, l2):
    # compute the norm of the difference of two lists or tuples.
    # only use the first two elements.
    return ((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)**(1/2) 


def generate_map(map_type:str):
    """
    Create a set of 20 landmarks forming the map.
    """
    global params, landmarks
    # key = integer ID. value = (x,y) position.
    landmarks = {}

    if map_type in ["demo", "demo_map"]:
        # force number of landmarks to match.
        params["NUM_LANDMARKS"] = len(demo_map.keys()) 
        landmarks = demo_map
    elif map_type in ["random", "rand"]:
        id = 0
        # randomly spread landmarks across the map.
        while len(landmarks.keys()) < params["NUM_LANDMARKS"]:
            pos = (2*params["MAP_BOUND"]*random() - params["MAP_BOUND"], 2*params["MAP_BOUND"]*random() - params["MAP_BOUND"])
            dists = [ norm(lm_pos, pos) < params["MIN_SEP"] for lm_pos in landmarks.values()]
            if True not in dists:
                landmarks[id] = pos
                id += 1
    elif map_type == "grid":
        # place landmarks on a grid filling the bounds.
        id = 0
        for r in np.arange(-params["MAP_BOUND"], params["MAP_BOUND"], params["GRID_STEP"]):
            for c in np.arange(-params["MAP_BOUND"], params["MAP_BOUND"], params["GRID_STEP"]):
                landmarks[id] = (r, c)
                id += 1
        # update number of landmarks used.
        params["NUM_LANDMARKS"] = id
    else:
        rospy.logerr("Invalid map_type provided.")
        exit()

    # Publish ground truth of map and veh pose for plotter.
    # rospy.loginfo("Waiting to publish ground truth for plotter.")
    # if we send it before the other nodes have initialized, they'll miss their one chance to get it.
    rospy.sleep(1)
    true_map_msg = Float32MultiArray()
    true_map_msg.data = sum([[id, landmarks[id][0], landmarks[id][1]] for id in landmarks.keys()], [])
    true_map_pub.publish(true_map_msg)

    # send the first odom command and meas to kick off the EKF.
    cmd_pub.publish(Vector3(x=0, y=0))
    # lm_pub.publish(Float32MultiArray(data=[]))
    

def get_cmd(msg):
    """
    Receive new odom command, propagate the true state forward,
    and publish the new measurements.
    """
    global x_v
    # add noise to command.
    d = msg.x + 2*params["V_00"]*random()-params["V_00"]
    hdg = msg.y + 2*params["V_11"]*random()-params["V_11"]
    # cap cmds within odom constraints.
    d = max(0, min(d, params["ODOM_D_MAX"]))
    hdg = max(-params["ODOM_TH_MAX"], min(hdg, params["ODOM_TH_MAX"]))
    # update true veh position given this odom cmd.
    x_v = [x_v[0] + d*cos(x_v[2]), x_v[1] + d*sin(x_v[2]), x_v[2] + hdg]

    # publish new true position for plotting node.
    true_msg = Vector3(x=x_v[0], y=x_v[1], z=x_v[2])
    true_pose_pub.publish(true_msg)

    # generate measurements given this new veh position.
    # determine which landmarks are visible.
    visible_landmarks = []
    for id in range(params["NUM_LANDMARKS"]):
        # compute vector from veh pos to lm.
        diff_vec = [landmarks[id][i] - x_v[i] for i in range(2)]
        # extract range and bearing.
        r = norm(landmarks[id], x_v)
        gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
        beta = remainder(gb - x_v[2], tau) # bearing rel to robot
        # check if this is visible to the robot.
        if r > params["RANGE_MAX"]:
            continue
        elif beta > params["FOV_MIN"] and beta < params["FOV_MAX"]:
            # within range and fov.
            visible_landmarks.append([id, r, beta])
    # add noise to all detections and publish measurement.
    lm_msg = Float32MultiArray()
    lm_msg.data = (sum([[visible_landmarks[i][0], 
                    visible_landmarks[i][1]+2*params["W_00"]*random()-params["W_00"], 
                    visible_landmarks[i][2]+2*params["W_11"]*random()-params["W_11"]] 
                    for i in range(len(visible_landmarks))], []))
    lm_pub.publish(lm_msg)

    # TODO publish LiDAR measurements for occ grid node.


def main():
    global lm_pub, pkg_path, true_map_pub, true_pose_pub, cmd_pub
    rospy.init_node('sim_node')

    # read map type from command line arg.
    if len(sys.argv) < 2 or sys.argv[1] == "-1":
        rospy.logwarn("map not provided to data_fwd_node. Using random.\nOptions for map type include [random, grid, demo_full]")
        map_type = "random"
    else:
        map_type = sys.argv[1]

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')
    # read params.
    read_params(pkg_path)

    # subscribe to odom commands.
    rospy.Subscriber("/odom", Vector3, get_cmd, queue_size=1)
    # publish first odom cmd to start it all off.
    cmd_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

    # publish landmark detections: [id1,r1,b1,...idN,rN,bN]
    lm_pub = rospy.Publisher("/landmark", Float32MultiArray, queue_size=1)

    # publish ground truth for the plotter.
    true_pose_pub = rospy.Publisher("/truth/veh_pose",Vector3, queue_size=1)
    true_map_pub = rospy.Publisher("/truth/landmarks",Float32MultiArray, queue_size=1)

    # create the map.
    generate_map(map_type)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass