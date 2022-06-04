#!/usr/bin/env python3

"""
Particle Filter driver node.
Localize mobile robot using a known map.
"""

import rospy
from pf import PF
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from data_pkg.msg import Command, PFState

# import params script.
import rospkg
import importlib.util, sys
spec = importlib.util.spec_from_file_location("module.name", rospkg.RosPack().get_path('data_pkg')+"/src/import_params.py")
module = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = module
spec.loader.exec_module(module)
# set params.
params = module.Config.params

############ GLOBAL VARIABLES ###################
pf = PF(params)
set_pub = None
# Most recent control cmd and landmark measurements.
# cmd = [fwd dist, heading], lm_meas = [id, range, bearing, ...]
cmd_queue = []; lm_meas_queue = []
#################################################

# main PF loop that uses monte carlo localization.
def mcl_iteration(event):
    global lm_meas_queue, cmd_queue
    # skip if there's no prediction or measurement yet, or if the pf hasn't been initialized.
    if not pf.is_init or len(cmd_queue) < 1 or len(lm_meas_queue) < 1:
        return
    # pop the next data off the queue.
    cmd = cmd_queue.pop(0)
    z = lm_meas_queue.pop(0)
    
    # iterate the filter.
    pf.iterate(cmd, z)

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
    # set the map for the pf.
    pf.MAP = MAP

def get_cmd(msg):
    # get control command.
    global cmd_queue
    cmd_queue.append([msg.fwd, msg.ang])

def get_landmarks(msg):
    # get measurement of landmarks.
    # format: [id1,r1,b1,...idN,rN,bN]
    global lm_meas_queue
    lm_meas_queue.append(msg.data)

def get_init_veh_pose(msg):
    global cur
    # get the vehicle's starting pose.
    pf.init_particle_set([msg.x, msg.y, msg.z])

def main():
    global set_pub
    rospy.init_node('pf_node')

    # subscribe to the initial veh pose.
    rospy.Subscriber("/truth/init_veh_pose",Vector3, get_init_veh_pose, queue_size=1)

    # subscribe to landmark detections: [id1,r1,b1,...idN,rN,bN]
    rospy.Subscriber("/landmark", Float32MultiArray, get_landmarks, queue_size=1)
    # subscribe to control commands.
    rospy.Subscriber("/command", Command, get_cmd, queue_size=1)

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
