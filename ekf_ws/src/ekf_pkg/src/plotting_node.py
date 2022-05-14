#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from matplotlib import pyplot as plt

############ GLOBAL VARIABLES ###################
counter = 0
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
    veh_x.append(msg.data[0])
    veh_y.append(msg.data[1])
    veh_th.append(msg.data[2])
    if len(msg.data) > 3:
        lm_x = [msg.data[i] for i in range(3,len(msg.data),2)]
        lm_y = [msg.data[i] for i in range(4,len(msg.data),2)]

    # plot everything.
    plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black")
    # plt.scatter(veh_x, veh_y, s=12, color="green")
    # plot only new pos.
    plt.scatter(msg.data[0], msg.data[1], s=12, color="green")

    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("EKF-Estimated Trajectory and Landmarks")
    plt.draw()
    plt.pause(0.00000000001)


def main():
    rospy.init_node('plotting_node')

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