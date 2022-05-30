#!/usr/bin/env python3

"""
Command vehicle to pursue path to goal point chosen by clicking the plot.
Generate the next odom command based on current state estimate.
"""
import rospy
from data_pkg.msg import Command
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from ekf_pkg.msg import EKFState
from pf_pkg.msg import PFState
from cv_bridge import CvBridge
from pure_pursuit import PurePursuit
from astar import Astar

# import params script.
import rospkg
import importlib.util, sys
spec = importlib.util.spec_from_file_location("module.name", rospkg.RosPack().get_path('data_pkg')+"/src/import_params.py")
module = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = module
spec.loader.exec_module(module)
# set params in sub-scripts.
params = module.Config.params
PurePursuit.params = params
Astar.params = params
# set rel neighbors list for A* search.
Astar.nbrs = [(0, -1), (0, 1), (-1, 0), (1, 0)] + [(-1, -1), (-1, 1), (1, -1), (1, 1)] if params["ASTAR_INCL_DIAGONALS"] else []

############ GLOBAL VARIABLES ###################
cmd_pub = None; path_pub = None
# current estimate of veh pos.
cur = [params["x_0"], params["y_0"], params["yaw_0"]]
#################################################

def get_ekf_state(msg):
    """
    Get the state published by the EKF.
    """
    # update current position.
    global cur
    cur = [msg.x_v, msg.y_v, msg.yaw_v]
    # check size of path.
    path_len = len(PurePursuit.goal_queue)
    # call the desired navigation function.
    if params["NAV_METHOD"] == "pp":
        # use pure pursuit.
        cmd_pub.publish(PurePursuit.get_next_cmd(cur))
    elif params["NAV_METHOD"] in ["direct", "simple"]:
        # directly go to each point.
        cmd_pub.publish(PurePursuit.direct_nav(cur))
    else:
        rospy.logerr("Invalid NAV_METHOD choice.")
        exit()
    # if the path length has changed (by a point being reached), update the plot.
    if len(PurePursuit.goal_queue) != path_len:
        path_pub.publish(Float32MultiArray(data=sum(PurePursuit.goal_queue, [])))


def get_goal_pt(msg):
    # verify chosen pt is on the map and not in collision.
    try:
        if occ_map[Astar.tf_ekf_to_map((msg.x,msg.y))[0]][Astar.tf_ekf_to_map((msg.x,msg.y))[1]] == 0:
            rospy.logerr("Invalid goal point (in collision).")
            return
    except:
        rospy.logerr("Selected point is outside map bounds.")
        return

    rospy.loginfo("Setting goal pt to ("+"{0:.4f}".format(msg.x)+", "+"{0:.4f}".format(msg.y)+")")
    # if running in "simple" mode, only add goal point rather than path planning.
    if params["NAV_METHOD"] == "simple":
        PurePursuit.goal_queue.append([msg.x, msg.y])
        return
    # determine starting pos for path.
    if len(PurePursuit.goal_queue) > 0: # use end of prev segment as start if there is one.
        start = PurePursuit.goal_queue[-1]
    else: # otherwise use the current position estimate.
        start = cur
    # generate path with A*.
    path = Astar.astar(start, [msg.x, msg.y])
    if path is None:
        rospy.logerr("No path found by A*.")
        return
    # turn this path into a list of positions for goal_queue.
    new_path_segment = Astar.interpret_astar_path(path)
    # add these to the pure pursuit path.
    PurePursuit.goal_queue += new_path_segment
    # publish this path for the plotter to display.
    path_pub.publish(Float32MultiArray(data=sum(PurePursuit.goal_queue, [])))


def get_occ_map(msg):
    global occ_map
    # get the true occupancy grid map image.
    bridge = CvBridge()
    occ_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # set map for A* to use.
    Astar.occ_map = occ_map


def main():
    global cmd_pub, path_pub
    rospy.init_node('goal_pursuit_node')

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    # rospy.Subscriber("/state/pf", PFState, get_pf_state, queue_size=1)
    # subscribe to the true occupancy grid.
    rospy.Subscriber("/truth/occ_grid", Image, get_occ_map, queue_size=1)

    # publish odom commands for the vehicle.
    cmd_pub = rospy.Publisher("/command", Command, queue_size=1)

    # subscribe to current goal point.
    rospy.Subscriber("/plan/goal", Vector3, get_goal_pt, queue_size=1)
    # publish planned path to the goal.
    path_pub = rospy.Publisher("/plan/path", Float32MultiArray, queue_size=1)


    # instruct the user.
    rospy.loginfo("Left-click on the plot to set the vehicle's goal position.")

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass