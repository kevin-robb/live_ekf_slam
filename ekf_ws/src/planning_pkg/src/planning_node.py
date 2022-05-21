#!/usr/bin/env python3

"""
Generate the next odom command based on current state estimate.
"""

import rospy
import rospkg
from geometry_msgs.msg import Vector3
from ekf_pkg.msg import EKFState
from pf_pkg.msg import PFState
from random import random
import numpy as np

############ GLOBAL VARIABLES ###################
params = {}
cmd_pub = None
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


def choose_command(state_msg):
    """
    Given the current state estimate,
    choose and send an odom cmd to the vehicle.
    TODO also get occ grid and do A* on it.
    """
    # for now just make at random.
    odom_msg = Vector3()
    # odom_msg.x = params["ODOM_D_MAX"]*random() # distance.
    # odom_msg.y = 2*params["ODOM_TH_MAX"]*random() - params["ODOM_TH_MAX"] # heading.
    odom_msg.x = 0.05
    odom_msg.y = 0.05 * np.sign(state_msg.x_v)
    cmd_pub.publish(odom_msg)



def get_ekf_state(msg):
    """
    get the state published by the EKF.
    """
    choose_command(msg)

def get_pf_state(msg):
    """
    get the state published by the PF.
    """
    # TODO
    pass
    
def main():
    global cmd_pub
    rospy.init_node('planning_node')

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')
    # read params.
    read_params(pkg_path)

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    rospy.Subscriber("/state/pf", PFState, get_pf_state, queue_size=1)
    # TODO subscribe to the occupancy grid.

    # publish odom commands for the vehicle.
    cmd_pub = rospy.Publisher("/odom", Vector3, queue_size=100)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass