#!/usr/bin/env python3

"""
Generate the next odom command based on current state estimate.
"""

import rospy
import rospkg
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from ekf_pkg.msg import EKFState
from pf_pkg.msg import PFState
from random import random
import numpy as np
from math import remainder, tau, atan2
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib

############ GLOBAL VARIABLES ###################
params = {}
cmd_pub = None
goal = [0.0, 0.0] # goal x,y point.
goal_plot = None
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
    odom_msg = Vector3()

    # compute vector from veh pos est to goal.
    diff_vec = [goal[0]-state_msg.x_v, goal[1]-state_msg.y_v]
    # extract range and bearing differences.
    r = norm([state_msg.x_v, state_msg.y_v], goal)
    gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
    beta = remainder(gb - state_msg.yaw_v, tau) # bearing rel to robot

    # turn these into commands.
    # go faster the more aligned the hdg is.
    odom_msg.x = 0.05 * (1 - abs(beta)/30)**5 + 0.005 if r > 0.05 else 0.0
    P = 0.03 #if r > 0.1 else 0.1
    odom_msg.y = beta * P

    cmd_pub.publish(odom_msg)

def set_cmd():
    global goal
    r = rospy.Rate(1/params["DT"])
    while not rospy.is_shutdown():
        inp = input("goal pos: ").split(" ")
        goal = [float(inp[0]), float(inp[1])]
        r.sleep()


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
    

def on_move(event):
    # get the x and y pixel coords
    x, y = event.x, event.y
    if event.inaxes:
        ax = event.inaxes  # the axes instance
        print('data coords %f %f' % (event.xdata, event.ydata))

def on_click(event):
    global goal, goal_plot
    if event.button is MouseButton.LEFT:
        # set clicked point to the new goal.
        print("Setting goal to ("+str(event.xdata)+", "+str(event.ydata)+")")
        goal = [event.xdata, event.ydata]
        # update the plot to show the current goal.
        if goal_plot is not None:
            goal_plot.remove()
        goal_plot = plt.scatter(event.xdata, event.ydata, color="yellow", edgecolors="black", s=30)
        # print('disconnecting callback')
        # plt.disconnect(binding_id)

def move_figure(f, x, y):
# change position that the plot appears.
# https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

def get_true_map(msg):
    """
    Show the user the true map to allow them to select a goal position.
    NOTE this true map is NOT used by the path planner; it relies only on the EKF state estimate.
    """
    # plot the true map to compare to estimates.
    lm_x = [msg.data[i] for i in range(1,len(msg.data),3)]
    lm_y = [msg.data[i] for i in range(2,len(msg.data),3)]
    true_map = [lm_x, lm_y]
    plt.scatter(true_map[0], true_map[1], s=30, color="white", edgecolors="black")

def main():
    global cmd_pub, binding_id
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

    # subscribe to landmark map for goal choice plot.
    rospy.Subscriber("/truth/landmarks",Float32MultiArray, get_true_map, queue_size=1)

    # ask user for commands.
    # set_cmd()
    # make sneaky plot behind the visible one.
    plt.rcParams["figure.figsize"] = (4,4)
    fig = plt.figure()
    move_figure(fig, 1400, 550)
    # set plot params to be the same as the real one.
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-1.5*params["MAP_BOUND"],1.5*params["MAP_BOUND"]])
    plt.ylim([-1.5*params["MAP_BOUND"],1.5*params["MAP_BOUND"]])
    plt.title("EKF-Estimated Trajectory and Landmarks")
    # register clicks on the plot.
    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)
    # "show" the plot.
    plt.show()

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass