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
#################################################

# get the state published by the EKF.
def get_state(msg):
    global veh_x, veh_y, veh_th, lm_x, lm_y, counter
    # rospy.loginfo("State: " + str(msg.data))
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

    # plot covariance for new pos.
    # referenced https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    # and https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot-python

    # got cov P_t from EKF. need to take only first 2x2 for position.
    cov = P_t[0:2,0:2]
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 0.025 * 2 * np.sqrt(vals)
    # draw parametric ellipse (instead of using patches).
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([w*np.cos(t) , h*np.sin(t)])
    R_rot = np.array([[cos(theta) , -sin(theta)],[sin(theta) , cos(theta)]])
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
    plt.plot(msg.data[9]+Ell_rot[0,:] , msg.data[10]+Ell_rot[1,:],'lightgrey' )

    # plot landmarks.
    plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black")
    # plot veh pos.
    # plt.scatter(veh_x, veh_y, s=12, color="green")
    plt.scatter(msg.data[9], msg.data[10], s=12, color="green")

    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("EKF-Estimated Trajectory and Landmarks")
    plt.draw()
    plt.pause(0.00000000001)


def save_plot():
    # save plot we've been building upon exit.
    # save to a file in pkg/plots directory.
    plt.savefig(pkg_path+"/plots/rss_test.png", format='png')

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

    # startup the plot.
    plt.ion()
    plt.show()

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass