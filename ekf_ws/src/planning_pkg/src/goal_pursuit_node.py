#!/usr/bin/env python3

"""
Command vehicle to pursue path to goal point chosen by clicking the plot.
Generate the next odom command based on current state estimate.
"""
import rospy
from base_pkg.msg import Command
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from base_pkg.msg import EKFState, UKFState #, PFState
from cv_bridge import CvBridge
from pure_pursuit import PurePursuit
from astar import Astar

# import params script.
import rospkg
import importlib.util, sys
spec = importlib.util.spec_from_file_location("module.name", rospkg.RosPack().get_path('base_pkg')+"/src/import_params.py")
module = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = module
spec.loader.exec_module(module)
# set params.
params = module.Config.params
PurePursuit.params = params
Astar.params = params
# set rel neighbors list for A* search.
Astar.nbrs = [(0, -1), (0, 1), (-1, 0), (1, 0)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if params["ASTAR_INCL_DIAGONALS"] else [])

############ GLOBAL VARIABLES ###################
cmd_pub = None; path_pub = None
# current estimate of veh pos.
cur = None
#################################################

def get_ekf_state(msg):
    """
    Get the state published by the EKF/UKF.
    """
    # update current position.
    global cur
    cur = [msg.x_v, msg.y_v, msg.yaw_v]
    if params["USE_LOCAL_PLANNER"] and msg.timestep % 5 == 0:
        # choose arbitrary free point ahead to use as goal.
        goal = Astar.local_planner(cur)
        if goal is None:
            rospy.logwarn("Could not find a free point to plan towards.")
            cmd_pub.publish(Command(fwd=0,ang=0))
            return
        # clear the current path.
        PurePursuit.goal_queue = []
        # generate path to get to new goal.
        find_path_to_goal(goal)

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
    # check operating mode.
    if params["USE_LOCAL_PLANNER"]:
        rospy.logerr("Cannot select a goal point while running in local planner mode.")
        return
    # verify chosen pt is on the map and not in collision.
    try:
        if occ_map[Astar.tf_ekf_to_map((msg.x,msg.y))[0]][Astar.tf_ekf_to_map((msg.x,msg.y))[1]] == 0:
            rospy.logerr("Invalid goal point (in collision).")
            return
    except:
        rospy.logerr("Selected point is outside map bounds.")
        return
    rospy.loginfo("Setting goal pt to ("+"{0:.4f}".format(msg.x)+", "+"{0:.4f}".format(msg.y)+")")
    find_path_to_goal([msg.x, msg.y])


def find_path_to_goal(goal):
    """
    Given a desired goal point, use A* to generate a path there.
    """
    # if running in "simple" mode, only add goal point rather than path planning.
    if params["NAV_METHOD"] == "simple" or params["BLANK_MAP"]:
        PurePursuit.goal_queue.append(goal)
        return
    # determine starting pos for path.
    if len(PurePursuit.goal_queue) > 0: # use end of prev segment as start if there is one.
        start = PurePursuit.goal_queue[-1]
    else: # otherwise use the current position estimate.
        start = cur
    # generate path with A*.
    path = Astar.astar(start, goal)
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

def get_init_veh_pose(msg):
    global cur
    # get the vehicle's starting pose.
    cur = [msg.x, msg.y, msg.z]

def main():
    global cmd_pub, path_pub, params
    rospy.init_node('goal_pursuit_node')

    # read command line args.
    if len(sys.argv) < 5:
        rospy.logerr("Required param (use_local_planner) and optional params (precompute_trajectory, tight_control, occ_map_img) not provided to goal_pursuit_node.")
        exit()
    else:
        params["USE_LOCAL_PLANNER"] = sys.argv[1].lower() == "true"
        precompute_trajectory = sys.argv[2].lower() == "true"
        params["TIGHT_CONTROL"] = sys.argv[3].lower() == "true"
        occ_map_img = sys.argv[4]
    # if traj was precomputed, this node is unnecessary.
    if precompute_trajectory:
        rospy.loginfo("goal_pursuit_node unneeded if trajectory was precomputed. Exiting.")
        exit()
    # if we're using the blank occ map, we don't need to do any A* planning.
    params["BLANK_MAP"] = occ_map_img == "blank.jpg"
    # set Pure Pursuit coefficients based on tight_control setting.
    if params["TIGHT_CONTROL"]:
        PurePursuit.get_control = PurePursuit.cmd_tight
    else:
        PurePursuit.get_control = PurePursuit.cmd_loose

    # subscribe to the initial veh pose.
    rospy.Subscriber("/truth/init_veh_pose",Vector3, get_init_veh_pose, queue_size=1)
    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    rospy.Subscriber("/state/ukf", UKFState, get_ekf_state, queue_size=1)
    # rospy.Subscriber("/state/pf", PFState, get_pf_state, queue_size=1)
    # subscribe to the true occupancy grid.
    rospy.Subscriber("/truth/occ_grid", Image, get_occ_map, queue_size=1)

    # publish odom commands for the vehicle.
    cmd_pub = rospy.Publisher("/command", Command, queue_size=1)

    # subscribe to current goal point.
    rospy.Subscriber("/plan/goal", Vector3, get_goal_pt, queue_size=1)
    # publish planned path to the goal.
    path_pub = rospy.Publisher("/plan/path", Float32MultiArray, queue_size=1)

    if not params["USE_LOCAL_PLANNER"]:
        # instruct the user to choose goal point(s).
        rospy.loginfo("Left-click on the plot to set the vehicle's goal position.")

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass