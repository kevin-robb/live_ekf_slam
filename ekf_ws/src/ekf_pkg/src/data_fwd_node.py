#!/usr/bin/env python3

"""
Test node to create/send odom and landmark measurements
to the EKF node for verification and debugging.
"""

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
import sys
from random import random
from math import pi, atan2, remainder, tau, cos, sin
import numpy as np

############ GLOBAL VARIABLES ###################
USE_RSS_DATA = False # T = use demo set. F = randomize map and create new traj.
DT = 0.5 # timer period used if cmd line param not provided.
odom_pub = None; lm_pub = None; true_map_pub = None; true_pose_pub = None
pkg_path = None # filepath to this package.
# Default map from RSS demo:
demo_map = { 0 : (6.2945, 8.1158), 1 : (-7.4603, 8.2675), 2 : (2.6472, -8.0492), 
        3 : (-4.4300, 0.9376), 4 : (9.1501, 9.2978), 5 : (-6.8477, 9.4119), 6 : (9.1433, -0.2925),
        7 : (6.0056, -7.1623), 8 : (-1.5648, 8.3147), 9 : (5.8441, 9.1898), 10: (3.1148, -9.2858),
        11: (6.9826, 8.6799), 12: (3.5747, 5.1548), 13: (4.8626, -2.1555), 14: (3.1096, -6.5763),
        15: (4.1209, -9.3633), 16: (-4.4615, -9.0766), 17: (-8.0574, 6.4692), 18: (3.8966, -3.6580), 19: (9.0044, -9.3111) }
#################################################

def main_loop(event):
    # send stuff to the EKF for testing.
    odom_msg = Vector3()
    odom_msg.x = 0.1
    odom_msg.y = 0.1
    odom_pub.publish(odom_msg)
    lm_msg = Float32MultiArray()
    lm_msg.data = [1, 0.5, 0.7, 4, 1.2, -0.9]
    lm_pub.publish(lm_msg)

# get the state published by the EKF.
def get_state(msg):
    # rospy.loginfo("State: " + str(msg.data[9:12]))
    pass


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

    # send data one timestep at a time to the ekf.
    i_z = 0; i = 0
    r = rospy.Rate(1/DT) # freq in Hz
    while not rospy.is_shutdown():
        if i == len(z_id): return
    
        odom_msg = Vector3()
        odom_msg.x = odom_dist[i]
        odom_msg.y = odom_hdg[i]
        odom_pub.publish(odom_msg)
        # only send landmarks when there is a detection (z_id != 0).
        if z_id[i] != 0:
            lm_msg = Float32MultiArray()
            lm_msg.data = [z_id[i], z_range[i_z], z_bearing[i_z]]
            lm_pub.publish(lm_msg)
            i_z += 1
        i += 1
        # sleep to publish at desired freq.
        r.sleep()


def norm(l1, l2):
    # compute the norm of the difference of two lists or tuples.
    # only use the first two elements.
    return ((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)**(1/2) 


def generate_data():
    """
    Create a set of 20 landmarks forming the map.
    Choose a trajectory through the space that will
    come near a reasonable amount of landmarks.
    Generate this trajectory's odom commands and
    measurements for all timesteps.
    Publish these one at a time for the EKF.
    """
    ################ GENERATE MAP #######################
    BOUND = 10 # all lm will be w/in +- BOUND in both x/y.
    NUM_LANDMARKS = 20
    MIN_SEP = 0.05 # min dist between landmarks.
    # key = integer ID. value = (x,y) position.
    landmarks = {}

    USE_DEMO_MAP = False
    if USE_DEMO_MAP:
        if len(demo_map.keys()) == NUM_LANDMARKS:
            landmarks = demo_map
        else:
            rospy.logwarn("NUM_LANDMARKS must match length of demo_map to use it.")
    # make new map if the demo hasn't been set.
    if len(landmarks.keys()) < 1:
        id = 0
        while len(landmarks.keys()) < NUM_LANDMARKS:
            pos = (2*BOUND*random() - BOUND, 2*BOUND*random() - BOUND)
            dists = [ norm(lm_pos, pos) < MIN_SEP for lm_pos in landmarks.values()]
            if True not in dists:
                landmarks[id] = pos
                id += 1
    
    ############# GENERATE TRAJECTORY ###################
    NUM_TIMESTEPS = 1000
    # constraints:
    ODOM_D_MAX = 0.1; ODOM_TH_MAX = 0.0546
    # process noise in EKF. will be used to add noise to odom.
    # V = np.array([[0.02**2,0.0],[0.0,(0.5*pi/180)**2]]) # deg
    # V = np.array([[0.02**2,0.0],[0.0,0.25**2]]) # rad
    V = np.array([[0.0,0.0],[0.0,0.0]]) # no noise
    # param to keep track of true current pos.
    x0 = [0.0,0.0,0.0]
    x_v = x0
    # init the odom lists.
    odom_dist = []; odom_hdg = []
    """
    We will use travelling salesman problem (TSP) - 
    nearest neighbors approach to find an efficient path.
    NOTE: We'll add noise to landmarks, but this method of
    trajectory generation assumes we have a rough idea of 
    where all landmarks are. The EKF doesn't get this info,
    so it is still performing fully blind EKF SLAM, but
    the trajectory planner is a knowing helper.
    """
    # make noisy version of rough map to use.
    LM_NOISE = 0.2
    noisy_lm = {}
    for id in range(NUM_LANDMARKS):
        noisy_lm[id] = (landmarks[id][0] + 2*LM_NOISE*random()-LM_NOISE, landmarks[id][1] + 2*LM_NOISE*random()-LM_NOISE)
    # choose nearest landmark to x0 as first node.
    cur_goal = 0; cur_dist = norm(noisy_lm[cur_goal], x_v)
    for id in range(NUM_LANDMARKS):
        if norm(noisy_lm[id], x_v) < cur_dist:
            cur_goal = id
            cur_dist = norm(noisy_lm[id], x_v)
    # build directed graph using NN heuristic.
    cur_node = cur_goal
    # store path of lm indices to visit in order.
    lm_path = [cur_node]
    unvisited = [id for id in range(NUM_LANDMARKS)]
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
    t = 0; THRESHOLD = 3
    for t in range(NUM_TIMESTEPS):
        # first entry in lm_path always the current goal.
        # we will move it to the end once approx achieved.
        # thus, robot will loop around until time runs out.
        if norm(x_v, noisy_lm[lm_path[0]]) < THRESHOLD:
            # mark as ~ arrived.
            lm_path = lm_path[1:] + [lm_path[0]]

        # move towards current goal.
        # compute vector from veh pos to lm.
        diff_vec = [noisy_lm[lm_path[0]][i] - x_v[i] for i in range(2)]
        # extract range and bearing.
        d = norm(noisy_lm[lm_path[0]], x_v)
        gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
        hdg = remainder(gb - x_v[2], tau) # bearing rel to robot.
        # choose odom cmd w/in constraints.
        d = min(d, ODOM_D_MAX) # always pos.
        if abs(hdg) > ODOM_TH_MAX:
            # cap magnitude but keep sign.
            hdg = ODOM_TH_MAX * np.sign(hdg)
        # update veh position given this odom cmd.
        x_v = [x_v[0] + d*cos(x_v[2]), x_v[1] + d*sin(x_v[2]), x_v[2] + hdg]
        # add noise to odom and add to trajectory.
        odom_dist.append(d + 2*V[0,0]*random()-V[0,0])
        odom_hdg.append(hdg + 2*V[1,1]*random()-V[1,1])

    #####################################################
    # Optionally replace odom with the demo for debugging.
    USE_DEMO_ODOM = False
    if USE_DEMO_ODOM:
        odom_file = open(pkg_path+"/data/ekf_odo_m.csv", "r")
        odom_raw = odom_file.readlines()
        odom_dist = [float(d) for d in odom_raw[0].split(",")]
        odom_hdg = [float(h) for h in odom_raw[1].split(",")]
    #####################################################
    # Propagate odom to get veh pos at all times.
    pos_true = [x0]; x_v = x0
    for t in range(NUM_TIMESTEPS):
        x_v = [x_v[0] + odom_dist[t]*cos(x_v[2]), x_v[1] + odom_dist[t]*sin(x_v[2]), x_v[2] + odom_hdg[t]]
        pos_true.append(x_v)
    ############# GENERATE MEASUREMENTS #################
    # vision constraints:
    RANGE_MAX = 4; FOV = [-pi, pi]
    # sensing noise in EKF.
    # W = np.array([[0.1**2,0.0],[0.0,(1*pi/180)**2]]) # deg
    # W = np.array([[0.1**2,0.0],[0.0,0.5**2]]) # rad
    W = np.array([[0.0,0.0],[0.0,0.0]]) # no noise
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
    for t in range(NUM_TIMESTEPS):
        # loop and determine which landmarks are visible.
        visible_landmarks = []
        for id in range(NUM_LANDMARKS):
            # compute vector from veh pos to lm.
            diff_vec = [landmarks[id][i] - pos_true[t][i] for i in range(2)]
            # extract range and bearing.
            r = norm(landmarks[id], pos_true[t])
            gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
            beta = remainder(gb - pos_true[t][2], tau) # bearing rel to robot
            # check if this is visible to the robot.
            if r > RANGE_MAX:
                continue
            elif beta > FOV[0] and beta < FOV[1]:
                # within range and fov.
                visible_landmarks.append([id, r, beta])
        # set number of detections on this timestep.
        z_num_det.append(len(visible_landmarks))
        # add noise to all detections and add to list.
        z.append(sum([[visible_landmarks[i][0], visible_landmarks[i][1]+2*W[0,0]*random()-W[0,0], visible_landmarks[i][2]+2*W[1,1]*random()-W[1,1]] for i in range(len(visible_landmarks))], []))

    ############### SEND DATA ################
    # Publish ground truth of map and veh pose for plotter.
    rospy.logwarn("Waiting to publish ground truth for plotter.")
    rospy.sleep(1)
    true_pose_msg = Float32MultiArray()
    true_pose_msg.data = sum(pos_true, [])
    true_pose_pub.publish(true_pose_msg)
    true_map_msg = Float32MultiArray()
    true_map_msg.data = sum([[id, landmarks[id][0], landmarks[id][1]] for id in landmarks.keys()], [])
    true_map_pub.publish(true_map_msg)
    # Send noisy data one timestep at a time to the ekf.
    t = 0
    r = rospy.Rate(1/DT) # freq in Hz
    while not rospy.is_shutdown():
        if t == NUM_TIMESTEPS: return
    
        odom_msg = Vector3()
        odom_msg.x = odom_dist[t]
        odom_msg.y = odom_hdg[t]
        odom_pub.publish(odom_msg)
        # always send a measurement. empty list if no detection.
        lm_msg = Float32MultiArray()
        lm_msg.data = z[t]
        lm_pub.publish(lm_msg)
        # increment time.
        t += 1
        # sleep to publish at desired freq.
        r.sleep()
    

def main():
    global lm_pub, odom_pub, pkg_path, DT, true_map_pub, true_pose_pub
    rospy.init_node('data_fwd_node')

    # read DT from command line arg.
    try:
        if len(sys.argv) < 2 or sys.argv[1] == "-1":
            rospy.logwarn("DT not provided to data_fwd_node. Using DT="+str(DT))
        else:
            DT = float(sys.argv[1])
            if DT <= 0:
                raise Exception("Negative DT issued.")
    except:
        rospy.logerr("DT param must be a positive float.")
        exit()

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ekf_pkg')

    # publish landmark detections: [id1,r1,b1,...idN,rN,bN]
    lm_pub = rospy.Publisher("/landmark", Float32MultiArray, queue_size=1)
    # publish odom commands/measurements.
    odom_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

    # publish ground truth for the plotter.
    true_pose_pub = rospy.Publisher("/truth/veh_pose",Float32MultiArray, queue_size=1)
    true_map_pub = rospy.Publisher("/truth/landmarks",Float32MultiArray, queue_size=1)

    # subscribe to the current state.
    rospy.Subscriber("/ekf/state", Float32MultiArray, get_state, queue_size=1)

    if USE_RSS_DATA:
        # read data from RSS ex4.
        read_rss_data()
    else:
        # create data in same format.
        generate_data()
        # run the main loop.
        # rospy.Timer(rospy.Duration(DT), main_loop)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
