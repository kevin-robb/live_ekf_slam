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

############ GLOBAL VARIABLES ###################
USE_RSS_DATA = True
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
    zind_file = open(pkg_path+"/data/ekf_zind_m.csv", "r")
    zind = [int(num) for num in zind_file.readlines()[0].split(",")]

    # send data one at a time to the ekf.
    i_z = 0; i = 0
    r = rospy.Rate(1/DT) # freq in Hz
    while not rospy.is_shutdown():
        if i == len(zind): return
    
        odom_msg = Vector3()
        odom_msg.x = odom_dist[i]
        odom_msg.y = odom_hdg[i]
        odom_pub.publish(odom_msg)
        # only send landmarks when there is a detection (zind != 0).
        if zind[i] != 0:
            lm_msg = Float32MultiArray()
            lm_msg.data = [zind[i], z_range[i_z], z_bearing[i_z]]
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
        if len(sys.argv) > 1:
            DT = float(sys.argv[1])
        else:
            rospy.logwarn("DT not provided to data_fwd_node. Using DT="+str(DT))
    except:
        print("DT param must be a float.")
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
        # use whatever test data is in main loop.
        rospy.Timer(rospy.Duration(DT), main_loop)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass