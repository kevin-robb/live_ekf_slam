#!/usr/bin/env python3

"""
This is the main EKF SLAM node.
The state will be [xV,yV,thV,x1,y1,...,xM,yM],
where the first three are the vehicle state,
and the rest are the positions of M landmarks.
"""

import rospy
import rospkg
import numpy as np
from math import sin, cos, remainder, tau, atan2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from ekf_pkg.msg import EKFState

############ GLOBAL VARIABLES ###################
params = {}
V = None; W = None
state_pub = None
############## NEEDED BY EKF ####################
# Initial vehicle state mean and covariance.
x0 = np.array([[0.0],[0.0],[0.0]])
P0 = np.array([[0.01**2,0.0,0.0],[0.0,0.01**2,0.0],[0.0,0.0,0.005**2]])
# Most recent odom reading and landmark measurements.
odom_queue = []; lm_meas_queue = []
# Current state mean and covariance.
x_t = x0; P_t = P0
# IDs of seen landmarks. Order corresponds to ind in state.
lm_IDs = []
# timestep number for debugging.
timestep = 0
#################################################

def read_params(pkg_path):
    """
    Read params from config file.
    @param path to data_pkg.
    """
    global params, V, W
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
    # set process and sensing noise.
    V = np.array([[params["V_00"],0.0],[0.0,params["V_11"]]])
    W = np.array([[params["W_00"],0.0],[0.0,params["W_11"]]])


# main EKF loop that happens every timestep.
def ekf_iteration(event):
    global x_t, P_t, lm_IDs, lm_meas_queue, odom_queue, timestep
    # skip if params not read yet or there's no prediction or measurement.
    if W is None or len(odom_queue) < 1 or len(lm_meas_queue) < 1:
        return
    timestep += 1
    # pop the next data off the queue.
    odom = odom_queue.pop(0)
    lm_meas = lm_meas_queue.pop(0)

    ############# PREDICTION STEP ##################
    # odom gives us a (dist, heading) "command".
    d_d = odom[0]; d_th = odom[1]

    # Compute jacobian matrices.
    F_xv = np.array([[1,0,-d_d*sin(x_t[2,0])],
                     [0,1,d_d*cos(x_t[2,0])],
                     [0,0,1]])
    F_x = np.eye(x_t.shape[0]) #* 0
    # F_x = np.zeros((x_t.shape[0],x_t.shape[0]))
    F_x[0:3,0:3] = F_xv
    F_vv = np.array([[cos(x_t[2,0]), 0],
                     [sin(x_t[2,0]),0],
                     [0,1]])
    F_v = np.zeros((x_t.shape[0],2))
    F_v[0:3,0:2] = F_vv

    # Make predictions.
    # landmarks are assumed constant, so we predict only vehicle position will change.
    x_pred = x_t
    x_pred[0,0] = x_t[0,0] + (d_d + params["v_d"])*cos(x_t[2,0])
    x_pred[1,0] = x_t[1,0] + (d_d + params["v_d"])*sin(x_t[2,0])
    x_pred[2,0] = x_t[2,0] + d_th + params["v_th"]
    # cap heading to (-pi,pi).
    x_pred[2,0] = remainder(x_pred[2,0], tau)
    # propagate covariance.
    P_pred = F_x @ P_t @ F_x.T + F_v @ V @ F_v.T

    ################## UPDATE STEP #######################
    # we only use landmark measurements, so skip if there aren't any.
    if len(lm_meas) > 0:
        # at least one landmark was detected since the last EKF iteration.
        # we can run the update step once for each landmark.
        num_landmarks = len(lm_meas) // 3
        for l in range(num_landmarks):
            # extract single landmark observation.
            id = int(lm_meas[l*3]); r = lm_meas[l*3 + 1]; b = lm_meas[l*3 + 2]
            # check if this is the first detection of this landmark ID.
            if id in lm_IDs:
                ################ LANDMARK UPDATE ###################
                # this landmark is already in our state, so update it.
                i = lm_IDs.index(id)*2 + 3 # index of lm x in state.

                # Compute Jacobian matrices.
                dist = ((x_t[i,0]-x_pred[0,0])**2 + (x_t[i+1,0]-x_pred[1,0])**2)**(1/2)
                H_xv = np.array([[-(x_t[i,0]-x_pred[0,0])/dist, -(x_t[i+1,0]-x_pred[1,0])/dist, 0], [(x_t[i+1,0]-x_pred[1,0])/(dist**2), -(x_t[i,0]-x_pred[0,0])/(dist**2), -1]])
                H_xp = np.array([[(x_t[i,0]-x_pred[0,0])/dist, (x_t[i+1,0]-x_pred[1,0])/dist], [-(x_t[i+1,0]-x_pred[1,0])/(dist**2), (x_t[i,0]-x_pred[0,0])/(dist**2)]])
                H_x = np.zeros((2,x_pred.shape[0]))
                H_x[0:2,0:3] = H_xv
                H_x[0:2,i:i+2] = H_xp
                H_w = np.eye(2)

                # Update the state and covariance.
                # compute innovation.
                ang = remainder(atan2(x_t[i+1,0]-x_pred[1,0], x_t[i,0]-x_pred[0,0])-x_pred[2,0],tau)
                z_est = np.array([[dist], [ang]])
                nu = np.array([[r],[b]]) - z_est - np.array([[params["w_r"]],[params["w_b"]]])
                # compute kalman gain.
                S = H_x @ P_pred @ H_x.T + H_w @ W @ H_w.T
                K = P_pred @ H_x.T @ np.linalg.inv(S)
                # perform update.
                x_pred = x_pred + K @ nu
                # cap heading to (-pi,pi).
                x_pred[2,0] = remainder(x_pred[2,0], tau)
                P_pred = P_pred - K @ H_x @ P_pred
            else :
                ############# LANDMARK INSERTION #######################
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
                G_x = np.array([[1, 0, -r*sin(x_pred[2,0]+b)],[0, 1, r*cos(x_pred[2,0]+b)]])
                # form insertion jacobian.
                Y = np.eye(n+2)
                Y[n:n+2,n:n+2] = G_z
                Y[n:n+2,0:3] = G_x

                # update covariance.
                P_pred = Y @ np.vstack([np.hstack([P_pred, np.zeros((n,2))]), np.hstack([np.zeros((2,n)), W])]) @ Y.T
                
    # finalize the state update. this works even if no landmarks were detected.
    x_t = x_pred; P_t = P_pred
    # cap heading to (-pi,pi).
    x_t[2,0] = remainder(x_t[2,0], tau)
    # publish the current cov + state.
    send_state()


# create EKFState message and publish it.
def send_state():
    msg = EKFState()
    msg.timestep = timestep
    msg.x_v = x_t[0,0]
    msg.y_v = x_t[1,0]
    msg.yaw_v = x_t[2,0]
    msg.M = (x_t.shape[0] - 3) // 2
    msg.landmarks = sum([[lm_IDs[i], x_t[3+2*i,0], x_t[3+2*i+1,0]] for i in range(len(lm_IDs))], [])
    # covariance change to one row.
    msg.P = []
    for subl in P_t.tolist():
        msg.P += subl
    msg.P += (x_t.T).tolist()[0]
    # publish it.
    state_pub.publish(msg)


# get measurement of odometry info.
def get_odom(msg):
    global odom_queue
    # format: x = distance, y = heading.
    odom_queue.append([msg.x, msg.y])

# get measurement of landmarks.
def get_landmarks(msg):
    # format: [id1,range1,bearing1,...idN,rN,bN]
    global lm_meas_queue
    lm_meas_queue.append(msg.data)

def main():
    global state_pub
    rospy.init_node('ekf_node')

    # find the filepath to data package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')
    # read params.
    read_params(pkg_path)

    # subscribe to landmark detections: [id1,r1,b1,...idN,rN,bN]
    rospy.Subscriber("/landmark", Float32MultiArray, get_landmarks, queue_size=1)
    # subscribe to odom commands/measurements.
    rospy.Subscriber("/odom", Vector3, get_odom, queue_size=1)

    # create publisher for the current state.
    state_pub = rospy.Publisher("/state/ekf", EKFState, queue_size=1)

    rospy.Timer(rospy.Duration(params["DT"]), ekf_iteration)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass