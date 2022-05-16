#!/usr/bin/env python3

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray
from matplotlib import pyplot as plt
import numpy as np
import atexit
from math import cos, sin, pi

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

############ GLOBAL VARIABLES ###################
# state data from the EKF to use for plotting.
# position/hdg for all times.
veh_x = []; veh_y = []; veh_th = []
# current estimate for all landmark positions.
lm_x = []; lm_y = []
# ground truth veh pose.
true_pose = None; true_map = None
# plotting stuff.
SHOW_ENTIRE_TRAJ = False
SHOW_TRUE_TRAJ = True
ARROW_LEN = 0.1
lm_pts = None; veh_pts = None; ell_pts = None; veh_true = None
#################################################

# get the state published by the EKF.
def get_state(msg):
    global veh_x, veh_y, veh_th, lm_x, lm_y, lm_pts, veh_pts, ell_pts, veh_true, true_map, true_pose
    # save what we want to plot.
    # first 9 represent the 3x3 covariance matrix.
    P_t = np.array([[msg.data[0],msg.data[1],msg.data[2]],
                    [msg.data[3],msg.data[4],msg.data[5]],
                    [msg.data[6],msg.data[7],msg.data[8]]])
    # rest is the state.
    veh_x.append(msg.data[9])
    veh_y.append(msg.data[10])
    veh_th.append(msg.data[11])
    if len(msg.data) > 3:
        lm_x = [msg.data[i] for i in range(12,len(msg.data),2)]
        lm_y = [msg.data[i] for i in range(13,len(msg.data),2)]

    # plot ground truth map.
    if true_map is not None:
        plt.scatter(true_map[0], true_map[1], s=30, color="white", edgecolors="black")
        true_map = None

    # plot full ground truth trajectory.
    if SHOW_TRUE_TRAJ and true_pose is not None:
        for timestep in range(0,len(true_pose)//3):
            plt.arrow(true_pose[timestep*3], true_pose[timestep*3+1], ARROW_LEN*cos(true_pose[timestep*3+2]), ARROW_LEN*sin(true_pose[timestep*3+2]), color="blue", width=0.01)
        true_pose = None


    # plot EKF covariance for current position.
    # referenced https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    # and https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot-python

    # got cov P_t from EKF. need to take only first 2x2 for position.
    cov = P_t[0:2,0:2]
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    num_std_dev = 2
    w, h = num_std_dev * 2 * np.sqrt(vals)
    # draw parametric ellipse (instead of using patches).
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([w*np.cos(t) , h*np.sin(t)])
    R_rot = np.array([[cos(theta) , -sin(theta)],[sin(theta) , cos(theta)]])
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
    # remove old ellipses.
    if not SHOW_ENTIRE_TRAJ and ell_pts is not None:
        ell_pts.remove()
    # plot the ellipse.
    ell_pts, = plt.plot(msg.data[9]+Ell_rot[0,:] , msg.data[10]+Ell_rot[1,:],'lightgrey')

    # remove old landmark estimates
    if lm_pts is not None:
        lm_pts.remove()
    # plot new landmark estimates.
    lm_pts = plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black")

    # plot current EKF-estimated veh pos.
    if not SHOW_ENTIRE_TRAJ and veh_pts is not None:
        veh_pts.remove()
    # draw a single pt with arrow to represent current veh pose.
    veh_pts = plt.arrow(msg.data[9], msg.data[10], ARROW_LEN*cos(msg.data[11]), ARROW_LEN*sin(msg.data[11]), color="green", width=0.1)

    # do the plotting.
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("EKF-Estimated Trajectory and Landmarks")
    plt.draw()
    plt.pause(0.00000000001)


def save_plot():
    # save plot we've been building upon exit.
    # save to a file in pkg/plots directory.
    # plt.savefig(pkg_path+"/plots/rss_test.png", format='png')
    pass


def get_true_pose(msg):
    # save the true traj to plot it along w cur state.
    global true_pose
    true_pose = msg.data


def get_true_map(msg):
    rospy.loginfo("Ground truth map received.")
    # plot the true map to compare to estimates.
    global true_map
    lm_x = [msg.data[i] for i in range(1,len(msg.data),3)]
    lm_y = [msg.data[i] for i in range(2,len(msg.data),3)]
    true_map = [lm_x, lm_y]
    

def main():
    global pkg_path
    rospy.init_node('plotting_node')

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ekf_pkg')

    # when the node exits, make the plot.
    atexit.register(save_plot)

    # subscribe to the current state.
    rospy.Subscriber("/ekf/state", Float32MultiArray, get_state, queue_size=1)

    # subscribe to ground truth.
    rospy.Subscriber("/truth/veh_pose",Float32MultiArray, get_true_pose, queue_size=1)
    rospy.Subscriber("/truth/landmarks",Float32MultiArray, get_true_map, queue_size=1)

    # startup the plot.
    plt.ion()
    plt.show()

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass