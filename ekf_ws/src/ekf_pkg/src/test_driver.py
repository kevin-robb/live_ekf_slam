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
import atexit

############ GLOBAL VARIABLES ###################
DT = 1 # timer period.
odom_pub = None
lm_pub = None
# state data from the EKF to use for plotting.
# position/hdg for all times.
veh_x = []; veh_y = []; veh_th = []
# current estimate for all landmark positions.
lm_x = []; lm_y = []
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
    global veh_x, veh_y, veh_th, lm_x, lm_y
    rospy.loginfo("State: " + str(msg.data))
    # save what we want to plot.
    veh_x.append(msg.data[0])
    veh_y.append(msg.data[1])
    veh_th.append(msg.data[2])
    if len(msg.data) > 3:
        lm_x = [msg.data[i] for i in range(3,len(msg.data),2)]
        lm_y = [msg.data[i] for i in range(4,len(msg.data),2)]

def read_rss_data():
    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ekf_pkg')
    # read odometry commands.
    odom_file = open(pkg_path+"/data/ekf_odo_m.csv", "r")
    odom_raw = odom_file.readlines()
    odom_dist = [float(d) for d in odom_raw[0].split(",")]
    odom_hdg = [float(h) for h in odom_raw[1].split(",")]
    # odom = [[float(odom_raw[0][i]),float(odom_raw[1][i])] for i in range(len(odom_raw[0]))]
    # read landmarks measurements.
    z_file = open(pkg_path+"/data/ekf_z_m.csv", "r")
    z_raw = z_file.readlines()
    z_range = [float(r) for r in z_raw[0].split(",")]
    z_bearing = [float(b) for b in z_raw[1].split(",")]
    # z = [[float(z_raw[0][i]),float(z_raw[1][i])] for i in range(len(z_raw[0]))]
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
        # update the plot.
        # make_plot()
        # sleep to publish at desired freq.
        r.sleep()

def make_plot():
    # plt.close()
    # extract all landmark positions.
    # lm_x = []; lm_y = []
    # for i in range(len(landmarks) // 2):
    #     lm_x.append(landmarks[i*2])
    #     lm_y.append(landmarks[i*2+1])

    plt.figure(figsize=(8,7))
    plt.grid(True)
    plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black")
    plt.scatter(veh_x, veh_y, s=12, color="green")

    # other plot formatting.
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Estimated Trajectory and Landmarks")
    plt.tight_layout()
    # save to a file in plots/ directory.
    # plt.savefig("./plots/"+FILE_ID+".png", format='png')
    # show the plot.
    # plt.show()
    # update the plot.
    plt.draw()

def main():
    global lm_pub, odom_pub
    rospy.init_node('test_driver_node')

    # when the node exits, make the plot.
    atexit.register(make_plot)

    # publish landmark detections: [id1,r1,b1,...idN,rN,bN]
    lm_pub = rospy.Publisher("/landmark/apriltag", Float32MultiArray, queue_size=1)
    # publish odom commands/measurements.
    odom_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

    # subscribe to the current state.
    rospy.Subscriber("/ekf/state", Float32MultiArray, get_state, queue_size=1)

    # startup the plot.
    # plt.ion()
    # plt.show()

    USE_RSS_DATA = True
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