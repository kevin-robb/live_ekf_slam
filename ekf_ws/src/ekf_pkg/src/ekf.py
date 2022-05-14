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
DT = 1 # timer period.
state_pub = None
############## NEEDED BY EKF ####################
# Process noise in (forward, angular). Assume zero mean.
v_d = 0; v_th = 0
V = np.array([[1.0,0.0],[0.0,1.0]])
# Sensing noise in (range, bearing). Assume zero mean.
w_r = 0; w_b = 0
W = np.array([[1.0,0.0],[0.0,1.0]])
# Initial vehicle state mean and covariance.
x0 = np.array([[0.0],[0.0],[0.0]])
P0 = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
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
    if odom is None:
        rospy.loginfo("No odom, skipping EKF loop.")
        return
    else:
        rospy.loginfo("Got odom, proceeding.")

    # odom gives us a (dist, heading) "command".
    d_d = odom[0]; d_th = odom[1]

    # Compute jacobian matrices.
    F_xv = np.array([[1,0,-d_d*sin(x_t[2,0])],
                     [0,1,d_d*cos(x_t[2,0])],
                     [0,0,1]])
    F_x = np.eye(x_t.shape[0])
    F_x[0:3,0:3] = F_xv
    F_vv = np.array([[cos(x_t[2,0]), 0],
                     [sin(x_t[2,0]),0],
                     [0,1]])
    F_v = np.zeros((x_t.shape[0],2))
    F_v[0:3,0:2] = F_vv

    # print("x_t: ", x_t)
    # print("odom: ", [d_d, d_th])
    # print("F_x: ", F_x)
    # print("F_v: ", F_v)

    # Make predictions.
    # landmarks are assumed constant, so we predict only vehicle position will change.
    x_pred = x_t
    x_pred[0,0] = x_t[0,0].item() + (d_d + v_d)*cos(x_t[2,0].item())
    x_pred[1,0] = x_t[1,0].item() + (d_d + v_d)*sin(x_t[2,0].item())
    x_pred[2,0] = x_t[2,0].item() + d_th + v_th
    # print("x_pred: ", x_pred)
    # predict covariance.
    P_pred = F_x @ P_t @ F_x.T + F_v @ V @ F_v.T

    # Update step. we use landmark measurements.
    if lm_meas is not None:
        # print("lm_meas: ", lm_meas)
        # at least one landmark was detected since the last EKF iteration.
        # we can run the update step once for each landmark.
        num_landmarks = len(lm_meas) // 3
        for l in range(num_landmarks):
            # extract single landmark observation.
            id = lm_meas[l*3]; r = lm_meas[l*3 + 1]; b = lm_meas[l*3 + 2]
            # check if this is the first detection of this landmark ID.
            if id in lm_IDs:
                # this landmark is already in our state, so update it.
                i = lm_IDs.index(id)*2 + 3 # index of lm x in state.
                # Compute Jacobian matrices.
                dist = ((x_t[i,0]-x_pred[0,0])**2 + (x_t[i+1,0]-x_pred[1,0])**2)**(1/2)
                H_xv = np.array([[-(x_t[i,0]-x_pred[0,0])/dist, -(x_t[i+1,0]-x_pred[1,0])/dist, 0], [(x_t[i+1,0]-x_pred[1,0])/(dist**2), -(x_t[i,0]-x_pred[0,0])/(dist**2), -1]])
                H_xp = np.array([[(x_t[i,0]-x_pred[0,0])/dist, (x_t[i+1,0]-x_pred[1,0])/dist], [-(x_t[i+1,0]-x_pred[1,0])/(dist**2), (x_t[i,0]-x_pred[0,0])/(dist**2)]])
                H_x = np.zeros((2,x_t.shape[0]))
                H_x[0:2,0:3] = H_xv
                H_x[0:2,i:i+2] = H_xp
                H_w = np.eye(2)
                # Update the state and covariance.
                # compute innovation.
                ang = remainder(atan2(x_t[i+1,0]-x_pred[1,0], x_t[i,0]-x_pred[0,0])-x_pred[2,0],tau)
                z_est = np.array([[dist], [ang]])
                nu = np.array([[r],[b]]) - z_est - np.array([[w_r],[w_b]])
                # compute kalman gain.
                S = H_x @ P_pred @ H_x.T + H_w @ W @ H_w.T
                K = P_pred @ H_x.T @ np.linalg.inv(S)
                # perform update.
                x_pred = x_pred + K @ nu
                P_pred = P_pred - K @ H_x @ P_pred
            else :
                # this is our first time detecting this landmark ID.
                n = x_pred.shape[0]
                # add the new landmark to our state.
                g = np.array([[x_pred[0,0] + r*cos(x_pred[2,0]+b)],
                              [x_pred[1,0] + r*sin(x_pred[2,0]+b)]])
                x_pred = np.vstack([x_pred, g])
                # add landmark ID to our list.
                lm_IDs.append(id)
                # Compute Jacobian matrices.
                G_z = np.array([[cos(x_pred[2,0]+b), -r*sin(x_pred[2,0]+b)],
                                [sin(x_pred[2,0]+b), r*cos(x_pred[2,0]+b)]])
                Y_z = np.eye(n+2)
                Y_z[n:n+2,n:n+2] = G_z
                # update covariance.
                # print("P_pred: ", P_pred)
                # print("zeros1: ", np.zeros((n,2)))
                # print("zeros2: ", np.zeros((2,n)))
                # print("W: ", W)
                P_pred = Y_z @ np.vstack([np.hstack([P_pred, np.zeros((n,2))]), np.hstack([np.zeros((2,n)), W])]) @ Y_z.T
                
    # officially update the state. this works even if no landmarks were detected.
    x_t = x_pred; P_t = P_pred
    # publish the current state.
    msg = Float32MultiArray(); msg.data = (x_t.T).tolist()[0]
    state_pub.publish(msg)

    # at the end of each iteration, we mark used measurements
    # to None so they won't be used again as if they were unique.
    odom = None; lm_meas = None


# get measurement of odometry info.
def get_odom(msg):
    global odom
    odom = [msg.x, msg.y]

# get measurement of landmarks.
def get_landmarks(msg):
    # format: [id1,r1,b1,...idN,rN,bN]
    global lm_meas
    lm_meas = msg.data

def main():
    global state_pub
    rospy.init_node('ekf_node')

    # subscribe to landmark detections: [id1,r1,b1,...idN,rN,bN]
    rospy.Subscriber("/landmark/apriltag", Float32MultiArray, get_landmarks, queue_size=1)
    # subscribe to odom commands/measurements.
    rospy.Subscriber("/odom", Vector3, get_odom, queue_size=1)

    # create publisher for the current state.
    state_pub = rospy.Publisher("/ekf/state", Float32MultiArray, queue_size=1)

    rospy.Timer(rospy.Duration(DT), ekf_iteration)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass