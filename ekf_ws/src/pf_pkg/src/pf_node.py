#!/usr/bin/env python3

"""
Particle Filter driver node.
Localize mobile robot using a known map.
"""

import rospy
import rospkg
from pf import PF
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from data_pkg.msg import PFState

############ GLOBAL VARIABLES ###################
params = {}
pf = None
set_pub = None
# Most recent odom reading and landmark measurements.
# odom = [dist, heading], lm_meas = [id, range, bearing, ...]
odom_queue = []; lm_meas_queue = []
#################################################


def read_params(pkg_path):
    """
    Read params from config file.
    @param path to data_pkg.
    """
    global params
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


# main PF loop that uses monte carlo localization.
def mcl_iteration(event):
    global lm_meas_queue, odom_queue
    # skip if there's no prediction or measurement yet, or if the pf hasn't been initialized.
    if pf is None or len(odom_queue) < 1 or len(lm_meas_queue) < 1:
        return
    # pop the next data off the queue.
    odom = odom_queue.pop(0)
    z = lm_meas_queue.pop(0)
    
    # iterate the filter.
    pf.iterate(odom, z)

    # publish the particle set for plotting.
    set_pub.publish(pf.get_state())


# Getters
def get_true_map(msg):
    # get the map and init the pf.
    rospy.loginfo("Ground truth map received by PF.")
    global pf
    lm_x = [msg.data[i] for i in range(1,len(msg.data),3)]
    lm_y = [msg.data[i] for i in range(2,len(msg.data),3)]
    MAP = {}
    for id in range(len(lm_x)):
        MAP[id] = (lm_x[id], lm_y[id])
    # initialize the particle set.
    pf = PF(params, MAP)

def get_odom(msg):
    # get measurement of odometry info.
    global odom_queue
    odom_queue.append([msg.x, msg.y])

def get_landmarks(msg):
    # get measurement of landmarks.
    # format: [id1,r1,b1,...idN,rN,bN]
    global lm_meas_queue
    lm_meas_queue.append(msg.data)

def main():
    global set_pub
    rospy.init_node('pf_node')

    # find the filepath to data package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')
    # read params.
    read_params(pkg_path)

    # subscribe to landmark detections: [id1,r1,b1,...idN,rN,bN]
    rospy.Subscriber("/landmark", Float32MultiArray, get_landmarks, queue_size=1)
    # subscribe to odom commands/measurements.
    rospy.Subscriber("/odom", Vector3, get_odom, queue_size=1)

    # subscribe to true map.
    rospy.Subscriber("/truth/landmarks",Float32MultiArray, get_true_map, queue_size=1)

    # create publisher for the current state.
    set_pub = rospy.Publisher("/state/pf", PFState, queue_size=1)

    rospy.Timer(rospy.Duration(params["DT"]), mcl_iteration)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
