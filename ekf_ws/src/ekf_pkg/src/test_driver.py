#!/usr/bin/env python3

"""
This is the main EKF SLAM node.
The state will be [xV,yV,thV,x1,y1,...,xM,yM],
where the first three are the vehicle state,
and the rest are the positions of M landmarks.
"""

import rospy
import numpy as np
from math import sin, cos, remainder, tau, atan2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3

############ GLOBAL VARIABLES ###################
DT = 5 # timer period.
odom_pub = None
lm_pub = None
#################################################

def main_loop(event):
    # send stuff to the EKF for testing.
    odom_msg = Vector3()
    odom_msg.x = 0.1
    odom_msg.y = 0.1
    odom_pub.publish(odom_msg)
    lm_msg = Float32MultiArray()
    lm_msg.data = [1, 0.5, 0.7, 4, 1.2, -0.9]

# get the state published by the EKF.
def get_state(msg):
    rospy.loginfo("State: " + str(msg.data))

def main():
    global lm_pub, odom_pub
    rospy.init_node('test_driver_node')

    # publish landmark detections: [id1,r1,b1,...idN,rN,bN]
    lm_pub = rospy.Publisher("/landmark/apriltag", Float32MultiArray, queue_size=1)
    # publish odom commands/measurements.
    odom_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

    # subscribe to the current state.
    rospy.Subscriber("/ekf/state", Float32MultiArray, get_state, queue_size=1)

    rospy.Timer(rospy.Duration(DT), main_loop)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass