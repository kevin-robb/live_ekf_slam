#!/usr/bin/env python3

"""
Test node to create/send odom and landmark measurements
to the EKF node for verification and debugging.
"""

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from matplotlib import pyplot as plt
import sys
from random import random
from math import pi
import numpy as np

############ GLOBAL VARIABLES ###################
USE_RSS_DATA = True # T = use demo set. F = randomize map and create new traj.
DT = 0.5 # timer period used if cmd line param not provided.
odom_pub = None
lm_pub = None
pkg_path = None # filepath to this package.
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
    rospy.loginfo("State: " + str(msg.data[9:12]))


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

    # send data one at a time to the ekf.
    send_data(odom_dist, odom_hdg, z_id, z_range, z_bearing)


def generate_data():
    """
    Create a set of 20 landmarks forming the map.
    Choose a trajectory through the space that will
    come near a reasonable amount of landmarks.
    Generate this trajectory's odom commands and
    measurements for all timesteps.
    Publish these one at a time for the EKF.
    """
    x0 = np.array([[0.0],[0.0],[0.0]])
    ################ GENERATE MAP #######################
    BOUND = 10 # all lm will be w/in +- BOUND in both x/y.
    NUM_LANDMARKS = 20
    MIN_SEP = 0.05 # min dist between landmarks.
    # key = integer ID. value = (x,y) position.
    landmarks = {}; id = 0
    while len(landmarks.keys() < NUM_LANDMARKS):
        x_pos = 2*BOUND*random() - BOUND
        y_pos = 2*BOUND*random() - BOUND
        dists = [ ((lm_pos[0]-x_pos)**2 + (lm_pos[1]-y_pos)**2)**(1/2) < MIN_SEP for lm_pos in landmarks.values()]
        if True not in dists:
            landmarks[id] = (x_pos, y_pos)
            id += 1
    ############# GENERATE TRAJECTORY ###################
    NUM_TIMESTEPS = 1000
    # constraints:
    ODOM_D_MAX = 0.1; ODOM_TH_MAX = 0.0546

    odom_dist = None; odom_hdg = None
    ############# GENERATE MEASUREMENTS #################
    # vision constraints:
    RANGE_MAX = 4; FOV = [-pi, pi]
    
    
    z_id = None; z_range = None; z_bearing = None
    #####################################################
    send_data(odom_dist, odom_hdg, z_id, z_range, z_bearing)


def send_data(odom_dist, odom_hdg, z_id, z_range, z_bearing):
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


def main():
    global lm_pub, odom_pub, pkg_path, DT
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
    lm_pub = rospy.Publisher("/landmark/apriltag", Float32MultiArray, queue_size=1)
    # publish odom commands/measurements.
    odom_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

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