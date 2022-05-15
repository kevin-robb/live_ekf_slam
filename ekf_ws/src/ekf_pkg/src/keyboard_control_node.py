#!/usr/bin/env python3

"""
Script to use arrow keys to control robot
and generate odom/measurements.
"""
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from math import cos, sin, pi, tau, remainder, atan2
import numpy as np
from random import random
import keyboard
import sys

########### GLOBAL VARIABLES ############
DT = 0.5
x0 = [0.0, 0.0, 0.0]
x_v = x0 # current true vehicle pose.
# odom constraints:
ODOM_D_MAX = 0.1; ODOM_TH_MAX = 0.0546
# vision constraints:
RANGE_MAX = 4; FOV = [-pi, pi]
# sensing noise in EKF.
# W = np.array([[0.1**2,0.0],[0.0,(1*pi/180)**2]])
W = np.array([[0.0,0.0],[0.0,0.0]]) # no noise
# map parameters.
BOUND = 10 # all lm will be w/in +- BOUND in both x/y.
NUM_LANDMARKS = 20
MIN_SEP = 0.05 # min dist between landmarks.
landmarks = None
# current "speed" params for smooth intuitive control.
fwd_speed = 0.0; ang_speed = 0.0
FWD_INCR = 0.005; ANG_INCR = 0.001
#########################################


def generate_map():
    """
    Create a set of landmarks forming the map.
    """
    # key = integer ID. value = (x,y) position.
    landmarks = {}
    id = 0
    while len(landmarks.keys()) < NUM_LANDMARKS:
        pos = (2*BOUND*random() - BOUND, 2*BOUND*random() - BOUND)
        dists = [((lm_pos[0]-pos[0])**2 + (lm_pos[1]-pos[1])**2)**(1/2) < MIN_SEP for lm_pos in landmarks.values()]
        if True not in dists:
            landmarks[id] = pos
            id += 1
    return landmarks


def generate_measurement(odom):
    """
    To create measurements, we morph the truth with noise
    from a known distribution, to emulate what a robot
    might actually measure.
    We will track the robot's true position, and use
    known sensor range and FOV to determine which
    landmarks are detected, and what the measured
    range, bearing should be.
    """
    global x_v
    # Propagate odom to get veh pos.
    x_v = [x_v[0] + odom[0]*cos(x_v[2]), x_v[1] + odom[0]*sin(x_v[2]), x_v[2] + odom[1]]

    # determine which landmarks are visible.
    visible_landmarks = []
    for id in range(NUM_LANDMARKS):
        # compute vector from veh pos to lm.
        diff_vec = [landmarks[id][i] - x_v[i] for i in range(2)]
        # extract range and bearing.
        r = (diff_vec[0]**2+diff_vec[1]**2)**(1/2)
        gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
        beta = remainder(gb - x_v[2], tau) # bearing rel to robot
        # check if this is visible to the robot.
        if r > RANGE_MAX:
            continue
        elif beta > FOV[0] and beta < FOV[1]:
            # within range and fov.
            visible_landmarks.append([id, r, beta])
    # add noise to all detections and return.
    return sum([[visible_landmarks[i][0], visible_landmarks[i][1]+2*W[0,0]*random()-W[0,0], visible_landmarks[i][2]+2*W[1,1]*random()-W[1,1]] for i in range(len(visible_landmarks))], [])


def generate_odom(pressed):
    """
    Use keyboard presses to create odom [dist,hdg]
    command to send to the EKF.
    """
    global fwd_speed, ang_speed
    # linear.
    if pressed["up arrow"]:
        fwd_speed += FWD_INCR
    elif pressed["down arrow"]:
        fwd_speed -= FWD_INCR
    else:
        # drift back down to 0.
        fwd_speed -= 0.5*FWD_INCR
    # cap w/in constraints.
    fwd_speed = min(fwd_speed, ODOM_D_MAX) # always pos.
    # angular.
    if pressed["right arrow"]:
        ang_speed += ANG_INCR
    elif pressed["left arrow"]:
        ang_speed -= ANG_INCR
    else:
        # drift back towards 0 from either side.
        ang_speed -= 0.5*ANG_INCR*np.sign(ang_speed)
    # cap w/in constraints.
    ang_speed = remainder(ang_speed - x_v[2], tau) # bearing rel to robot.
    if abs(ang_speed) > ODOM_TH_MAX:
        # cap magnitude but keep sign.
        ang_speed = ODOM_TH_MAX * np.sign(ang_speed)
    # return odom command for this timestep.
    return [fwd_speed, ang_speed]


def record_keypress():
    """
    Record presses of the up, down, left, and right arrows.
    """
    pressed = {"left arrow" : False, "up arrow" : False, "down arrow" : False, "right arrow" : False}
    
    r = rospy.Rate(1/DT) # freq in Hz
    while not rospy.is_shutdown():
        for key in pressed.keys():
            pressed[key] = keyboard.is_pressed(key)
        # generate odom for current configuration.
        odom = generate_odom(pressed)
        # generate measurement.
        z = generate_measurement(odom)
        # publish these.
        odom_msg = Vector3(); odom_msg.x = odom[0]; odom_msg.y = odom[1]; odom_pub.publish(odom_msg)
        z_msg = Float32MultiArray(); z_msg.data = z; lm_pub.publish(z_msg)
        # wait to keep on track with desired frequency.
        r.sleep()


def main():
    global lm_pub, odom_pub, DT, true_map_pub, landmarks #, true_pose_pub
    rospy.init_node('keyboard_control_node')

    # read DT from command line arg.
    try:
        if len(sys.argv) < 2 or sys.argv[1] == "-1":
            rospy.logwarn("DT not provided to data_fwd_node. Using DT="+str(DT))
        else:
            DT = float(sys.argv[1])
            if DT <= 0:
                raise Exception("Negative DT issued.")
    except:
        rospy.logerr("DT param must be a positive float.")
        exit()

    # publish landmark detections: [id1,r1,b1,...idN,rN,bN]
    lm_pub = rospy.Publisher("/landmark", Float32MultiArray, queue_size=1)
    # publish odom commands/measurements.
    odom_pub = rospy.Publisher("/odom", Vector3, queue_size=1)

    # publish ground truth for the plotter.
    # true_pose_pub = rospy.Publisher("/truth/veh_pose",Float32MultiArray, queue_size=1)
    true_map_pub = rospy.Publisher("/truth/landmarks",Float32MultiArray, queue_size=1)

    # create the map.
    landmarks = generate_map()
    # publish true map for the plotter.
    rospy.sleep(0.5)
    true_map_msg = Float32MultiArray()
    true_map_msg.data = sum([[id, landmarks[id][0], landmarks[id][1]] for id in landmarks.keys()], [])
    true_map_pub.publish(true_map_msg)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
