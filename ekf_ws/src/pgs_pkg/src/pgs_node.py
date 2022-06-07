#!/usr/bin/env python3

"""
Use commands and measurements to perform Pose-Graph SLAM.
This estimates the map as well as all current and pase veh poses.
"""
import rospy
from data_pkg.msg import Command
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from data_pkg.msg import PGSState

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
state_pub = None
cmd_queue = []; lm_meas_queue = []
pgs = None
#################################################

def pgs_iteration(event):
    """
    Run one timestep of pose-graph slam using the next cmd and measurement.
    """
    # skip if params not read yet or there's no prediction or measurement.
    if pgs is None or len(cmd_queue) < 1 or len(lm_meas_queue) < 1:
        return
    # pop the next data off the queue.
    cmd = cmd_queue.pop(0)
    lm_meas = lm_meas_queue.pop(0)

def get_init_veh_pose(msg):
    global cur
    # get the vehicle's starting pose.
    cur = [msg.x, msg.y, msg.z]

def get_cmd(msg):
    # get commanded velocities.
    global cmd_queue
    cmd_queue.append([msg.fwd, msg.ang])

def get_landmarks(msg):
    # get measurement of landmarks.
    # format: [id1,range1,bearing1,...idN,rN,bN]
    global lm_meas_queue
    lm_meas_queue.append(msg.data)

def main():
    global state_pub
    rospy.init_node('pgs_node')

    # subscribe to the initial veh pose.
    rospy.Subscriber("/truth/init_veh_pose",Vector3, get_init_veh_pose, queue_size=1)
    # subscribe to commands sent to the vehicle.
    rospy.Subscriber("/command", Command, get_cmd, queue_size=1)
    # subscribe to landmark detections.
    rospy.Subscriber("/landmark", Float32MultiArray, get_landmarks, queue_size=1)

    # create publisher for the current state.
    state_pub = rospy.Publisher("/state/ekf", PGSState, queue_size=1)

    rospy.Timer(rospy.Duration(params["DT"]), pgs_iteration)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass