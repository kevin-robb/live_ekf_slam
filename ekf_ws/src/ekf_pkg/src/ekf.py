#!/usr/bin/env python3

"""
This is the main EKF SLAM node.
The state will be [xV,yV,thV,x1,y1,...,xM,yM],
where the first three are the vehicle state,
and the rest are the positions of M landmarks.
"""

import rospy
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray

############ GLOBAL VARIABLES ###################
DT = 1 # timer period.
state_pub = None
##########################################

def get_odom(msg):
    # TODO get measurement of odometry info.
    pass

def get_landmarks(msg):
    # get measurement of landmarks.
    # format: [id1,r1,b1,...idN,rN,bN]
    pass

def main():
    global tag_pub
    rospy.init_node('tag_tracking_node')

    # subscribe to landmark detections: [id1,r1,b1,...idN,rN,bN]
    rospy.Subscriber("/landmark/apriltag", Float32MultiArray, get_landmarks, queue_size=1)

    # create publisher for the current state.
    state_pub = rospy.Publisher("/ekf/state", Float32MultiArray, queue_size=1)

    # rospy.Timer(rospy.Duration(DT), timer_callback)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass