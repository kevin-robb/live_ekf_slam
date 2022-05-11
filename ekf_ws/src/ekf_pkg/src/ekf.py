#!/usr/bin/env python3

"""
This is the main EKF SLAM node.
The state will be [xV,yV,thV,x1,y1,...,xM,yM],
where the first three are the vehicle state,
and the rest are the positions of M landmarks.
"""

import rospy
import numpy as np
from math import sin, cos
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray

############ GLOBAL VARIABLES ###################
DT = 1 # timer period.
state_pub = None
############## NEEDED BY EKF ####################
# Process noise in (forward, angular). Assume zero mean.
v_d = 0; v_th = 0
V = np.array([[1,0],[0,1]])
# Sensing noise in (range, bearing). Assume zero mean.
w_r = 0; w_b = 0
W = np.array([[1,0],[0,1]])
# Initial vehicle state mean and covariance.
x0 = np.array([[0],[0],[0]])
P0 = np.array([[1,0,0],[0,1,0],[0,0,1]])
# Most recent odom reading.
odom = None # [dist, heading]
lm_meas = None # [id, range, bearing, ...]
# Current state mean and covariance.
x_t = x0; P_t = P0
# IDs of seen landmarks. Order corresponds to ind in state.
lm_IDs = []
#################################################

# main EKF loop that happens every timestep
def ekf_iteration(event):
    global odom, lm_meas, x_t, P_t, lm_IDs
    # skip if there's no prediction.
    if odom is None: return
    # odom gives us a (dist, heading) "command".
    d_d = odom[0]; d_th = odom[1]

    # Compute jacobian matrices.
    F_xv = np.array([[1,0,-d_d*sin(x_t[0,2])],
                     [0,1,d_d*cos(x_t[0,2])],
                     [0,0,1]])
    F_x = np.eye(x_t.shape[1])
    F_x[0:3,0:3] = F_xv
    F_vv = np.array([[cos(x_t[0,2]), 0],
                     [sin(x_t[0,2]),0],
                     [0,1]])
    F_v = np.zeros((x_t.shape[1],2))
    F_v[0:3,0:2] = F_vv

    # Make predictions.
    # landmarks are assumed constant, so we predict only vehicle position will change.
    x_pred = x_t
    x_pred[0,0] = x_t[0,0] + (d_d + v_d)*cos(x_t[0,2])
    x_pred[0,1] = x_t[0,1] + (d_d + v_d)*sin(x_t[0,2])
    x_pred[0,2] = x_t[0,2] + d_th + v_th
    # predict covariance.
    P_pred = F_x @ P_t @ F_x.T + F_v @ V @ F_v.T

    # Update step. we use landmark measurements.
    if lm_meas is not None:
        # at least one landmark was detected since the last EKF iteration.
        num_landmarks = len(lm_meas) // 3
        for i in range(num_landmarks):
            r = lm_meas[i*3 + 1]; b = lm_meas[i*3 + 2]
            # TODO stopped here

    # officially update the state. this works even if no landmarks were detected.
    x_t = x_pred; P_t = P_pred
    # publish the current state.
    msg = Float32MultiArray(); msg.data = x_t
    state_pub.publish(msg)

    # at the end of each iteration, we mark used measurements
    # to None so they won't be used again as if they were unique.
    odom = None; lm_meas = None



def get_odom(msg):
    # TODO get measurement of odometry info.
    pass

def get_landmarks(msg):
    # get measurement of landmarks.
    # format: [id1,r1,b1,...idN,rN,bN]
    pass

def main():
    global state_pub
    rospy.init_node('tag_tracking_node')

    # subscribe to landmark detections: [id1,r1,b1,...idN,rN,bN]
    rospy.Subscriber("/landmark/apriltag", Float32MultiArray, get_landmarks, queue_size=1)
    # TODO subscribe to odom commands/measurements.

    # create publisher for the current state.
    state_pub = rospy.Publisher("/ekf/state", Float32MultiArray, queue_size=1)

    rospy.Timer(rospy.Duration(DT), ekf_iteration)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass