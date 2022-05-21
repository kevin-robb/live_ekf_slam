#!/usr/bin/env python3

"""
Command vehicle to pursue path to goal point chosen by clicking the plot.
Generate the next odom command based on current state estimate.
"""

import rospy
import rospkg
from geometry_msgs.msg import Vector3
from ekf_pkg.msg import EKFState
from pf_pkg.msg import PFState
from math import remainder, tau, atan2

############ GLOBAL VARIABLES ###################
params = {}
cmd_pub = None
goal = [0.0, 0.0] # target x,y point.
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

def norm(l1, l2):
    # compute the norm of the difference of two lists or tuples.
    # only use the first two elements.
    return ((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)**(1/2) 

def choose_command(state_msg):
    """
    Given the current state estimate,
    choose and send an odom cmd to the vehicle.
    TODO also get occ grid and do A* on it.
    """
    global goal
    # check how close veh is to our current goal pt.
    r = norm([state_msg.x_v, state_msg.y_v], goal)
    # compute vector from veh pos est to goal.
    diff_vec = [goal[0]-state_msg.x_v, goal[1]-state_msg.y_v]
    # calculate bearing difference.
    gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
    beta = remainder(gb - state_msg.yaw_v, tau) # bearing rel to robot
    # turn this into a command.
    odom_msg = Vector3()
    # go faster the more aligned the hdg is.
    odom_msg.x = 0.05 * (1 - abs(beta)/30)**5 + 0.005 if r > 0.05 else 0.0
    P = 0.03 if r > 0.2 else 0.2
    odom_msg.y = beta * P if r > 0.05 else 0.0
    # ensure commands are capped within constraints.
    odom_msg.x = max(0, min(odom_msg.x, params["ODOM_D_MAX"]))
    odom_msg.y = max(-params["ODOM_TH_MAX"], min(odom_msg.y, params["ODOM_TH_MAX"]))
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
    
def get_goal_pt(msg):
    global goal
    goal = [msg.x, msg.y]
    rospy.loginfo("Setting goal pt to ("+"{0:.4f}".format(msg.x)+", "+"{0:.4f}".format(msg.y)+")")


def main():
    global cmd_pub
    rospy.init_node('goal_pursuit_node')

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
    cmd_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

    # subscribe to current goal point.
    rospy.Subscriber("/plan/goal", Vector3, get_goal_pt, queue_size=1)

    # instruct the user.
    rospy.loginfo("Left-click on the plot to set the vehicle's goal position.")

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass