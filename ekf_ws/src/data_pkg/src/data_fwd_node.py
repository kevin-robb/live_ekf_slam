#!/usr/bin/env python3

"""
Node to create/send odom and landmark measurements
to the EKF node for verification and demonstration.
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
DT = 0.05 # timer period used if cmd line param not provided.
odom_pub = None; lm_pub = None; true_map_pub = None; true_pose_pub = None
pkg_path = None # filepath to this package.
# Default map from RSS demo:
demo_map = { 0 : (6.2945, 8.1158), 1 : (-7.4603, 8.2675), 2 : (2.6472, -8.0492), 
        3 : (-4.4300, 0.9376), 4 : (9.1501, 9.2978), 5 : (-6.8477, 9.4119), 6 : (9.1433, -0.2925),
        7 : (6.0056, -7.1623), 8 : (-1.5648, 8.3147), 9 : (5.8441, 9.1898), 10: (3.1148, -9.2858),
        11: (6.9826, 8.6799), 12: (3.5747, 5.1548), 13: (4.8626, -2.1555), 14: (3.1096, -6.5763),
        15: (4.1209, -9.3633), 16: (-4.4615, -9.0766), 17: (-8.0574, 6.4692), 18: (3.8966, -3.6580), 19: (9.0044, -9.3111) }
#################################################


def read_params(pkg_path):
    """
    Read params from config file.
    @param path to data_pkg.
    """
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
        try:
            params[key] = int(arg)
        except:
            try:
                params[key] = float(arg)
            except:
                params[key] = (arg == "True")


def read_rss_data():
    """
    Files have a demo set of pre-generated data,
    including odometry commands and sensor measurements
    for all timesteps. Data association is known
    by giving landmark ID along with range/bearing.
    """
    # read odometry commands.
    odom_file = open(pkg_path+"/data/ekf_odo_m.csv", "r")
    odom_raw = odom_file.readlines()
    odom_dist = [float(d) for d in odom_raw[0].split(",")]
    odom_hdg = [float(h) for h in odom_raw[1].split(",")]
    # read landmarks measurements.
    z_file = open(pkg_path+"/data/ekf_z_m.csv", "r")
    z_raw = z_file.readlines()
    z_range = [float(r) for r in z_raw[0].split(",")]
    z_bearing = [float(b) for b in z_raw[1].split(",")]
    # read measured landmark indices.
    z_id_file = open(pkg_path+"/data/ekf_zind_m.csv", "r")
    z_id = [int(num) for num in z_id_file.readlines()[0].split(",")]

    # send the true map.
    # Publish ground truth of map and veh pose for plotter.
    rospy.loginfo("Waiting to publish ground truth for plotter.")
    # if we send it before the other nodes have initialized, they'll miss their one chance to get it.
    rospy.sleep(1)
    true_map_msg = Float32MultiArray()
    true_map_msg.data = sum([[id, demo_map[id][0], demo_map[id][1]] for id in demo_map.keys()], [])
    true_map_pub.publish(true_map_msg)
    # send data one timestep at a time to the ekf.
    i_z = 0; i = 0
    r = rospy.Rate(1/DT) # freq in Hz
    while not rospy.is_shutdown():
        if i == len(z_id): return
    
        odom_msg = Vector3()
        odom_msg.x = odom_dist[i]
        odom_msg.y = odom_hdg[i]
        odom_pub.publish(odom_msg)

        lm_msg = Float32MultiArray()
        lm_msg.data = []
        if z_id[i] != 0:
            lm_msg.data = [z_id[i]-1, z_range[i_z], z_bearing[i_z]]
            i_z += 1
        lm_pub.publish(lm_msg)
        i += 1
        # sleep to publish at desired freq.
        r.sleep()


def norm(l1, l2):
    # compute the norm of the difference of two lists or tuples.
    # only use the first two elements.
    return ((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)**(1/2) 


def generate_data(map_type:str):
    """
    Create a set of 20 landmarks forming the map.
    Choose a trajectory through the space that will
    come near a reasonable amount of landmarks.
    Generate this trajectory's odom commands and
    measurements for all timesteps.
    Publish these one at a time for the EKF.
    """
    global params
    ################ GENERATE MAP #######################
    # key = integer ID. value = (x,y) position.
    landmarks = {}

    if map_type == "demo_map":
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
    
    ############# GENERATE ODOMETRY COMMANDS ###################
    if map_type == "demo_odom":
        # use trajectory from demo instead of generating one.
        odom_file = open(pkg_path+"/data/ekf_odo_m.csv", "r")
        odom_raw = odom_file.readlines()
        odom_dist = [float(d) for d in odom_raw[0].split(",")]
        odom_hdg = [float(h) for h in odom_raw[1].split(",")]
    else:
        # param to keep track of true current pos.
        x0 = [0.0,0.0,0.0]
        # randomize starting pose.
        # x0 = [2*params["MAP_BOUND"]*random() - params["MAP_BOUND"], 2*params["MAP_BOUND"]*random() - params["MAP_BOUND"], 2*pi*random() - pi]
        x_v = x0
        # init the odom lists.
        odom_dist = []; odom_hdg = []
        """
        We will use travelling salesman problem (TSP) - 
        nearest neighbors approach to find a path.
        We'll add noise to landmarks, but this method of
        trajectory generation assumes we have a rough idea of 
        where all landmarks are. The EKF doesn't get this info,
        so it is still performing fully blind EKF-SLAM.
        """
        # make noisy version of rough map to use.
        noisy_lm = {}
        for id in range(params["NUM_LANDMARKS"]):
            noisy_lm[id] = (landmarks[id][0] + 2*params["LM_NOISE"]*random()-params["LM_NOISE"], landmarks[id][1] + 2*params["LM_NOISE"]*random()-params["LM_NOISE"])
        # choose nearest landmark to x0 as first node.
        cur_goal = 0; cur_dist = norm(noisy_lm[cur_goal], x_v)
        for id in range(params["NUM_LANDMARKS"]):
            if norm(noisy_lm[id], x_v) < cur_dist:
                cur_goal = id
                cur_dist = norm(noisy_lm[id], x_v)
        # build directed graph using NN heuristic.
        cur_node = cur_goal
        # store path of lm indices to visit in order.
        lm_path = [cur_node]
        unvisited = [id for id in range(params["NUM_LANDMARKS"])]
        unvisited.remove(cur_node)
        # find next nearest neighbor until all nodes are visited.
        while len(unvisited) > 0:
            cur_dist = -1
            # find nearest node to current one.
            for id in unvisited:
                dist = norm(noisy_lm[id], noisy_lm[cur_node])
                if cur_dist < 0 or dist < cur_dist:
                    cur_goal = id; cur_dist = dist
            # add this as next node to visit.
            lm_path.append(cur_goal)
            cur_node = cur_goal
            # mark it as visited.
            unvisited.remove(cur_node)
        # now traverse our graph to get an actual trajectory.
        t = 0
        for t in range(params["NUM_TIMESTEPS"]):
            # first entry in lm_path always the current goal.
            # we will move it to the end once approx achieved.
            # thus, robot will loop around until time runs out.
            if norm(x_v, noisy_lm[lm_path[0]]) < params["VISITATION_THRESHOLD"]:
                # mark as arrived.
                lm_path = lm_path[1:] + [lm_path[0]]

            # move towards current goal.
            # compute vector from veh pos to lm.
            diff_vec = [noisy_lm[lm_path[0]][i] - x_v[i] for i in range(2)]
            # extract range and bearing.
            d = norm(noisy_lm[lm_path[0]], x_v)
            gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
            hdg = remainder(gb - x_v[2], tau) # bearing rel to robot.
            # choose odom cmd w/in constraints.
            d = min(d, params["ODOM_D_MAX"]) # always pos.
            if abs(hdg) > params["ODOM_TH_MAX"]:
                # cap magnitude but keep sign.
                hdg = params["ODOM_TH_MAX"] * np.sign(hdg)
            # add noise.
            d = d + 2*params["V_00"]*random()-params["V_00"]
            hdg = hdg + 2*params["V_11"]*random()-params["V_11"]
            # update veh position given this odom cmd.
            x_v = [x_v[0] + d*cos(x_v[2]), x_v[1] + d*sin(x_v[2]), x_v[2] + hdg]
            # add noise to odom and add to trajectory.
            odom_dist.append(d)
            odom_hdg.append(hdg)

    ############## ODOM -> TRAJECTORY ##################
    # Propagate odom to get veh pos at all times.
    pos_true = [x0]; x_v = x0
    for t in range(params["NUM_TIMESTEPS"]):
        x_v = [x_v[0] + odom_dist[t]*cos(x_v[2]), x_v[1] + odom_dist[t]*sin(x_v[2]), x_v[2] + odom_hdg[t]]
        pos_true.append(x_v)

    ############# GENERATE MEASUREMENTS #################
    # init the meas lists.
    z_num_det = []; z = []
    """
    To create measurements, we morph the truth with noise
    from a known distribution, to emulate what a robot
    might actually measure.
    We will track the robot's true position, and use
    known sensor range and FOV to determine which
    landmarks are detected, and what the measured
    range, bearing should be.
    """
    for t in range(params["NUM_TIMESTEPS"]):
        # loop and determine which landmarks are visible.
        visible_landmarks = []
        for id in range(params["NUM_LANDMARKS"]):
            # compute vector from veh pos to lm.
            diff_vec = [landmarks[id][i] - pos_true[t][i] for i in range(2)]
            # extract range and bearing.
            r = norm(landmarks[id], pos_true[t])
            gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
            beta = remainder(gb - pos_true[t][2], tau) # bearing rel to robot
            # check if this is visible to the robot.
            if r > params["RANGE_MAX"]:
                continue
            elif beta > params["FOV_MIN"] and beta < params["FOV_MAX"]:
                # within range and fov.
                visible_landmarks.append([id, r, beta])
        # set number of detections on this timestep.
        z_num_det.append(len(visible_landmarks))
        # add noise to all detections and add to list.
        z.append(sum([[visible_landmarks[i][0], 
            visible_landmarks[i][1]+2*params["W_00"]*random()-params["W_00"], 
            visible_landmarks[i][2]+2*params["W_11"]*random()-params["W_11"]] 
            for i in range(len(visible_landmarks))], []))

    ############### SEND DATA ################
    # Publish ground truth of map and veh pose for plotter.
    rospy.loginfo("Waiting to publish ground truth for plotter.")
    # if we send it before the other nodes have initialized, they'll miss their one chance to get it.
    rospy.sleep(1)
    true_pose_msg = Float32MultiArray()
    true_pose_msg.data = sum(pos_true, [])
    true_pose_pub.publish(true_pose_msg)
    true_map_msg = Float32MultiArray()
    true_map_msg.data = sum([[id, landmarks[id][0], landmarks[id][1]] for id in landmarks.keys()], [])
    true_map_pub.publish(true_map_msg)
    # Send noisy data one timestep at a time to the ekf.
    t = 0
    odom_msg = Vector3()
    lm_msg = Float32MultiArray()
    r = rospy.Rate(1/DT) # freq in Hz
    while not rospy.is_shutdown():
        if t == params["NUM_TIMESTEPS"]: return
        # send odom as x,y components of a vector.
        odom_msg.x = odom_dist[t]
        odom_msg.y = odom_hdg[t]
        odom_pub.publish(odom_msg)
        # always send a measurement. empty list if no detection.
        lm_msg.data = z[t]
        lm_pub.publish(lm_msg)
        # increment time.
        t += 1
        # sleep to publish at desired freq.
        r.sleep()
    

def main():
    global lm_pub, odom_pub, pkg_path, DT, true_map_pub, true_pose_pub
    rospy.init_node('data_fwd_node')

    # read DT and map from command line arg.
    try:
        # get map type.
        if len(sys.argv) < 3 or sys.argv[2] == "-1":
            rospy.logwarn("map not provided to data_fwd_node. Using random.\nOptions for map type include [random, grid, demo_full]")
            map_type = "random"
        else:
            map_type = sys.argv[2]
        
        # get dt.
        if len(sys.argv) < 2 or sys.argv[1] == "-1":
            rospy.logwarn("DT not provided to data_fwd_node. Using DT="+str(DT))
        else:
            DT = float(sys.argv[1])
            if DT <= 0:
                raise Exception("DT must be positive.")
    except:
        rospy.logerr("DT param must be a positive float.")
        exit()

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')
    # read params.
    read_params(pkg_path)

    # publish landmark detections: [id1,r1,b1,...idN,rN,bN]
    lm_pub = rospy.Publisher("/landmark", Float32MultiArray, queue_size=1)
    # publish odom commands/measurements.
    odom_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

    # publish ground truth for the plotter.
    true_pose_pub = rospy.Publisher("/truth/veh_pose",Float32MultiArray, queue_size=1)
    true_map_pub = rospy.Publisher("/truth/landmarks",Float32MultiArray, queue_size=1)

    if map_type == "demo_full":
        # read data from RSS-RVC demo.
        read_rss_data()
    else:
        # create data.
        generate_data(map_type)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
