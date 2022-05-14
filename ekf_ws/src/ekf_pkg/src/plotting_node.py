#!/usr/bin/env python3

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray
from matplotlib import pyplot as plt
import atexit

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


def save_plot():
    # save plot to file upon exit.
    plt.figure(figsize=(8,7))
    plt.grid(True)
    plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black")
    plt.scatter(veh_x, veh_y, s=12, color="green")

    # other plot formatting.
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("EKF-Estimated Trajectory and Landmarks")
    plt.tight_layout()
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