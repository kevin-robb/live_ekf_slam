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
params = module.Config.params
# set params in sub-scripts.
PurePursuit.params = params
Astar.params = params

############ GLOBAL VARIABLES ###################
cmd_pub = None; path_pub = None
cur = [0.0, 0.0] # current estimate of veh pos.
lm_estimates = [] # current estimate of all landmark positions.
occ_map = None # cv2 image of true occupancy grid map.
#################################################
# Color codes for console output.
ANSI_RESET = "\u001B[0m"
ANSI_GREEN = "\u001B[32m"
ANSI_YELLOW = "\u001B[33m"
ANSI_BLUE = "\u001B[34m"
ANSI_PURPLE = "\u001B[35m"
ANSI_CYAN = "\u001B[36m"
#################################################



def get_ekf_state(msg):
    """
    Get the state published by the EKF.
    """
    # update current position estimates.
    global cur, lm_estimates
    cur = [msg.x_v, msg.y_v, msg.yaw_v]
    lm_estimates = msg.landmarks
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
    # update the map's landmark occ mask to influence path planning.
    # Astar.update_lm_mask(lm_estimates)

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


def get_occ_grid_map(msg):
    global occ_map
    # get the true occupancy grid map image.
    bridge = CvBridge()
    occ_map_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # create matrix from map, converting cells to int 0 (occluded) or 1 (free).
    # also expand the size of the map to fill the whole plot with a buffer of free space.
    num_buffer_rows = int(params["DISPLAY_REGION_BUFFER"] * params["OCC_MAP_SIZE"])
    occ_map = [[1] * num_buffer_rows] * num_buffer_rows
    offset = (num_buffer_rows - params["OCC_MAP_SIZE"]) // 2
    rospy.logwarn("num_rows, buffer, offset: "+str(len(occ_map_img[0]))+", "+str(num_buffer_rows)+", "+str(offset))
    # for i in range(len(occ_map_img)):
    #     for j in range(len(occ_map_img[0])):
    #         occ_map[offset+i][offset+j] = int(occ_map_img[i][j])
    # print(ANSI_YELLOW+"raw map:\n"+str(occ_map)+ANSI_RESET)

    # determine index pairs to select all neighbors when ballooning obstacles.
    nbrs = []
    for i in range(-params["OCC_MAP_BALLOON_AMT"], params["OCC_MAP_BALLOON_AMT"]+1):
        for j in range(-params["OCC_MAP_BALLOON_AMT"], params["OCC_MAP_BALLOON_AMT"]+1):
            nbrs.append((i, j))
    # set this to be used by A* for landmark avoidance.
    Astar.nbr_indexes = nbrs
    # remove 0,0 here which is just the parent cell.
    nbrs.remove((0,0))
    # expand all occluded cells outwards.
    for i in range(len(occ_map_img)):
        for j in range(len(occ_map_img[0])):
            # check original map for occlusion.
            if occ_map_img[i][j] < 0.5: # occluded.
                # mark all neighbors as occluded.
                for chg in nbrs:
                    occ_map[max(0, min(offset+i+chg[0], params["OCC_MAP_SIZE"]-1))][max(0, min(offset+j+chg[1], params["OCC_MAP_SIZE"]-1))] = 0
    # print(ANSI_BLUE+"inflated map:\n"+str(occ_map)+ANSI_RESET)
    # show value distribution in occ_map.
    freqs = [0, 0]
    for i in range(len(occ_map)):
        for j in range(len(occ_map[0])):
            if occ_map[i][j] == 0:
                freqs[0] += 1
            else:
                freqs[1] += 1
    print(ANSI_CYAN+"Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded."+ANSI_RESET)
    # set map for A* to use.
    Astar.occ_map = occ_map
    # create blank landmark occupancy mask at the right size.
    Astar.lm_mask = [[1] * len(occ_map)] * len(occ_map)


def main():
    global cmd_pub, path_pub
    rospy.init_node('goal_pursuit_node')

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    # rospy.Subscriber("/state/pf", PFState, get_pf_state, queue_size=1)
    # subscribe to the true occupancy grid.
    rospy.Subscriber("/truth/occ_grid", Image, get_occ_grid_map, queue_size=1)

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