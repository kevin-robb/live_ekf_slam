#!/usr/bin/env python3

"""
Node to track true state and send landmark measurements
to the EKF node, as well as LiDAR measurements to
the mapping node for occupancy grid creation.
Receives odom commands and performs them with noise.
"""

import rospy
from base_pkg.msg import Command
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
import sys
from random import random
from math import atan2, remainder, tau, cos, sin, pi
import numpy as np
import cv2
from cv_bridge import CvBridge
from import_params import Config

############ GLOBAL VARIABLES ###################
# publishers
lm_pub = None; true_map_pub = None; true_pose_pub = None; cmd_pub = None; color_map_pub = None
# Default map from RSS demo:
demo_map = { 0 : (6.2945, 8.1158), 1 : (-7.4603, 8.2675), 2 : (2.6472, -8.0492), 
        3 : (-4.4300, 0.9376), 4 : (9.1501, 9.2978), 5 : (-6.8477, 9.4119), 6 : (9.1433, -0.2925),
        7 : (6.0056, -7.1623), 8 : (-1.5648, 8.3147), 9 : (5.8441, 9.1898), 10: (3.1148, -9.2858),
        11: (6.9826, 8.6799), 12: (3.5747, 5.1548), 13: (4.8626, -2.1555), 14: (3.1096, -6.5763),
        15: (4.1209, -9.3633), 16: (-4.4615, -9.0766), 17: (-8.0574, 6.4692), 18: (3.8966, -3.6580), 19: (9.0044, -9.3111) }
# true landmark map and current pose.
landmarks = None; x_v = [0.0, 0.0, 0.0]
#################################################

def norm(l1, l2):
    # compute the norm of the difference of two lists or tuples.
    # only use the first two elements.
    return ((l1[0]-l2[0])**2 + (l1[1]-l2[1])**2)**(1/2) 


def tf_ekf_to_map(pt):
        # transform x,y from ekf coords to occ map indices.
        return [int(Config.params["SHIFT"] - pt[1] / Config.params["SCALE"]), int(Config.params["SHIFT"] + pt[0] / Config.params["SCALE"])]


def init_sim(occ_map_img:str, landmark_map:str, precompute_trajectory:bool):
    """
    Initialize the simulation based on launchfile params.
    """
    # read in the occupancy map.
    read_occ_map(occ_map_img)
    # generate the landmark map.
    generate_landmarks(landmark_map)

    if not precompute_trajectory:
        # send the first odom command and meas to kick off the EKF.
        cmd_pub.publish(Command(fwd=0, ang=0))
    else:
        # generate the entire trajectory, and publish all commands on a timer.
        generate_full_trajectory()


def generate_full_trajectory():
    """
    Precompute the entire trajectory for all timesteps.
    """
    ############# GENERATE ODOMETRY COMMANDS ###################
    # param to keep track of true current pos.
    x_t = x_v
    # init the odom lists.
    odom_fwd = []; odom_ang = []
    """
    We will use travelling salesman problem (TSP) - 
    nearest neighbors approach to find a path.
    We'll add noise to landmarks, but this method of
    trajectory generation assumes we have a rough idea of 
    where all landmarks are. The EKF doesn't get this info,
    so it is still performing fully blind EKF-SLAM.
    """
    # make noisy version of rough map to use.
    noisy_lm = {}
    for id in range(Config.params["NUM_LANDMARKS"]):
        noisy_lm[id] = (landmarks[id][0] + 2*Config.params["LM_NOISE"]*random()-Config.params["LM_NOISE"], 
                        landmarks[id][1] + 2*Config.params["LM_NOISE"]*random()-Config.params["LM_NOISE"])
        # keep all points well within the visible map area.
        noisy_lm[id] = (max(Config.params["DISPLAY_REGION"][0] + 1, min(noisy_lm[id][0], Config.params["DISPLAY_REGION"][1] - 1)),
                        max(Config.params["DISPLAY_REGION"][0] + 1, min(noisy_lm[id][1], Config.params["DISPLAY_REGION"][1] - 1)))
    # choose nearest landmark to x0 as first node.
    cur_goal = 0; cur_dist = norm(noisy_lm[cur_goal], x_t)
    for id in range(Config.params["NUM_LANDMARKS"]):
        if norm(noisy_lm[id], x_t) < cur_dist:
            cur_goal = id
            cur_dist = norm(noisy_lm[id], x_t)
    # build directed graph using NN heuristic.
    cur_node = cur_goal
    # store path of lm indices to visit in order.
    lm_path = [cur_node]
    unvisited = [id for id in range(Config.params["NUM_LANDMARKS"])]
    unvisited.remove(cur_node)
    # find next nearest neighbor until all nodes are visited.
    while len(unvisited) > 0:
        cur_dist = -1
        # find nearest node to current one.
        for id in unvisited:
            dist = norm(noisy_lm[id], noisy_lm[cur_node])
            if cur_dist < 0 or dist < cur_dist:
                cur_goal = id; cur_dist = dist
        # add this as next node to visit.
        lm_path.append(cur_goal)
        cur_node = cur_goal
        # mark it as visited.
        unvisited.remove(cur_node)
    # now traverse our graph to get an actual trajectory.
    for t in range(Config.params["NUM_TIMESTEPS"]):
        # first entry in lm_path always the current goal.
        # we will move it to the end once approx achieved.
        # thus, robot will loop around until time runs out.
        if norm(x_t, noisy_lm[lm_path[0]]) < Config.params["VISITATION_THRESHOLD"]:
            # mark as arrived.
            lm_path = lm_path[1:] + [lm_path[0]]

        # move towards current goal.
        # compute vector from veh pos to lm.
        diff_vec = [noisy_lm[lm_path[0]][i] - x_t[i] for i in range(2)]
        # extract range and bearing.
        d = norm(noisy_lm[lm_path[0]], x_t)
        gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
        hdg = remainder(gb - x_t[2], tau) # bearing rel to robot.
        # choose odom cmd w/in constraints.
        d = min(d, Config.params["ODOM_D_MAX"]) # always pos.
        if abs(hdg) > Config.params["ODOM_TH_MAX"]:
            # cap magnitude but keep sign.
            hdg = Config.params["ODOM_TH_MAX"] * np.sign(hdg)
        # update veh position given this odom cmd.
        x_t = [x_t[0] + d*cos(x_t[2]), x_t[1] + d*sin(x_t[2]), x_t[2] + hdg]
        # add noise to odom and add to trajectory.
        odom_fwd.append(d)
        odom_ang.append(hdg)

    ############### SEND DATA ################
    # Send control commands one timestep at a time to the ekf.
    t = 0
    r = rospy.Rate(1/Config.params["DT"]) # freq in Hz
    while not rospy.is_shutdown():
        if t == Config.params["NUM_TIMESTEPS"]: return
        # send odom command and true pose for plotting.
        cmd_pub.publish(Command(fwd=odom_fwd[t], ang=odom_ang[t]))
        # cmd_pub.publish(Command(fwd=0.1, ang=0.1))
        # increment time.
        t += 1
        # sleep to publish at desired freq.
        r.sleep()


def generate_landmarks(map_type:str):
    """
    Create a set of landmarks forming the map.
    """
    global landmarks
    # key = integer ID. value = (x,y) position.
    landmarks = {}

    if map_type == "demo":
        # force number of landmarks to match.
        Config.params["NUM_LANDMARKS"] = len(demo_map.keys()) 
        landmarks = demo_map
    elif map_type == "grid":
        # place landmarks on a grid filling the bounds.
        shift = Config.params["GRID_STEP"] / 2
        id = 0
        for r in np.arange(-Config.params["MAP_BOUND"]+shift, Config.params["MAP_BOUND"], Config.params["GRID_STEP"]):
            for c in np.arange(-Config.params["MAP_BOUND"]+shift, Config.params["MAP_BOUND"], Config.params["GRID_STEP"]):
                landmarks[id] = (r, c)
                id += 1
        # update number of landmarks used.
        Config.params["NUM_LANDMARKS"] = id
    elif map_type in ["random", "rand"]:
        id = 0
        # randomly spread landmarks across the map.
        while len(landmarks.keys()) < Config.params["NUM_LANDMARKS"]:
            pos = (2*Config.params["MAP_BOUND"]*random() - Config.params["MAP_BOUND"], 2*Config.params["MAP_BOUND"]*random() - Config.params["MAP_BOUND"])
            # check that it's not colliding with an obstacle.
            if occ_map[tf_ekf_to_map(pos)[0]][tf_ekf_to_map(pos)[1]] < 0.5: continue
            # check that it's not too close to any existing landmarks.
            if any([ norm(lm_pos, pos) < Config.params["MIN_SEP"] for lm_pos in landmarks.values()]): continue
            # add the landmark.
            landmarks[id] = pos
            id += 1
    elif map_type == "igvc1":
        barrels = [(8.16017316017316, -8.037518037518037), (7.727272727272725, -5.324675324675325), (8.419913419913419, -2.813852813852815), (8.910394265232974, -2.6695526695526706), (5.909090909090908, -1.2842712842712842), (6.457431457431456, -1.0822510822510836), (7.813852813852813, 0.3318903318903317), (6.688311688311687, 2.4675324675324664), (8.679653679653677, 5.064935064935064), (7.3232323232323235, 6.68109668109668), (8.535353535353535, 8.239538239538238), (5.995670995670993, 9.393939393939394), (0.7720057720057714, 5.728715728715727), (0.7142857142857135, 5.20923520923521), (2.7633477633477614, 4.458874458874458), (2.445887445887445, 4.141414141414142), (1.1183261183261166, 2.871572871572871), (0.916305916305916, 2.525252525252524), (2.5901875901875897, 1.9480519480519476), (2.6767676767676765, -3.795093795093795), (0.9740259740259738, -3.679653679653681), (-0.7287157287157289, -4.978354978354979), (-3.1818181818181834, -4.7186147186147185), (-2.129032258064516, -2.121212121212121), (-3.4992784992784998, -0.6493506493506498), (-1.5656565656565675, 1.5440115440115427), (-1.2770562770562783, 2.4098124098124085), (-2.0274170274170285, 3.9971139971139955), (-1.5079365079365097, 4.1991341991342), (-4.451659451659452, 4.805194805194805), (-7.9148629148629155, 3.1024531024531026), (-7.597402597402598, 1.0533910533910529), (-7.1067821067821075, 0.9668109668109661), (-7.53968253968254, -2.092352092352092), (-7.251082251082252, -4.054834054834055), (-9.040404040404042, -5.440115440115441), (-7.04906204906205, -7.373737373737375)]
        # update number of landmarks used.
        Config.params["NUM_LANDMARKS"] = len(barrels)
        # set the landmarks.
        for id in range(len(barrels)):
            landmarks[id] = barrels[id]
    else:
        rospy.logerr("Invalid map_type provided.")
        exit()

    # Publish ground truth of map and veh pose for plotter.
    # rospy.loginfo("Waiting to publish ground truth for plotter.")
    # if we send it before the other nodes have initialized, they'll miss their one chance to get it.
    # rospy.sleep(1)
    true_map_msg = Float32MultiArray()
    true_map_msg.data = sum([[id, landmarks[id][0], landmarks[id][1]] for id in landmarks.keys()], [])
    true_map_pub.publish(true_map_msg)
    

def get_cmd(msg):
    """
    Receive new odom command, propagate the true state forward,
    and publish the new measurements.
    """
    global x_v
    # add noise to command.
    d = msg.fwd + 2*Config.params["V_00"]*random()-Config.params["V_00"]
    hdg = msg.ang + 2*Config.params["V_11"]*random()-Config.params["V_11"]
    # cap cmds within odom constraints.
    d = max(0, min(d, Config.params["ODOM_D_MAX"]))
    hdg = max(-Config.params["ODOM_TH_MAX"], min(hdg, Config.params["ODOM_TH_MAX"]))
    # update true veh position given this odom cmd.
    x_v = [x_v[0] + d*cos(x_v[2]), x_v[1] + d*sin(x_v[2]), x_v[2] + hdg]

    # publish new true position for plotting node.
    true_msg = Vector3(x=x_v[0], y=x_v[1], z=x_v[2])
    true_pose_pub.publish(true_msg)

    # generate measurements given this new veh position.
    # determine which landmarks are visible.
    visible_landmarks = []
    for id in range(Config.params["NUM_LANDMARKS"]):
        # compute vector from veh pos to lm.
        diff_vec = [landmarks[id][i] - x_v[i] for i in range(2)]
        # extract range and bearing.
        r = norm(landmarks[id], x_v)
        gb = atan2(diff_vec[1], diff_vec[0]) # global bearing.
        beta = remainder(gb - x_v[2], tau) # bearing rel to robot
        # check if this is visible to the robot.
        if r > Config.params["RANGE_MAX"]:
            continue
        elif beta > Config.params["FOV_MIN"] and beta < Config.params["FOV_MAX"]:
            # within range and fov.
            visible_landmarks.append([id, r, beta])
    # add noise to all detections and publish measurement.
    lm_msg = Float32MultiArray()
    lm_msg.data = (sum([[visible_landmarks[i][0], 
                    visible_landmarks[i][1]+2*Config.params["W_00"]*random()-Config.params["W_00"], 
                    visible_landmarks[i][2]+2*Config.params["W_11"]*random()-Config.params["W_11"]] 
                    for i in range(len(visible_landmarks))], []))
    lm_pub.publish(lm_msg)

    # TODO publish LiDAR measurements for occ grid node.


def read_occ_map(occ_map_img:str):
    """
    Read in the map from the image file and convert it to an occupancy grid.
    Save a color map for display, and save a binarized occupancy grid for path planning.
    """
    global occ_map
    # read map image and account for possible white = transparency that cv2 will think is black.
    # https://stackoverflow.com/questions/31656366/cv2-imread-and-cv2-imshow-return-all-zeros-and-black-image/62985765#62985765
    img = cv2.imread(Config.params["BASE_PKG_PATH"]+'/config/maps/'+occ_map_img, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4: # we have an alpha channel
        a1 = ~img[:,:,3] # extract and invert that alpha
        img = cv2.add(cv2.merge([a1,a1,a1,a1]), img) # add up values (with clipping)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) # strip alpha channels
    # cv2.imshow('initial map', img); cv2.waitKey(0); cv2.destroyAllWindows()

    # save the color map for the plotter.
    # convert from BGR to RGB for display.
    color_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # lower the image resolution to the desired grid size.
    img = cv2.resize(img, (Config.params["OCC_MAP_SIZE"], Config.params["OCC_MAP_SIZE"]))

    # turn this into a grayscale img and then to a binary map.
    occ_map_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)[1]
    # normalize to range [0,1].
    occ_map_img = np.divide(occ_map_img, 255)
    # cv2.imshow("Thresholded Map", occ_map_img); cv2.waitKey(0); cv2.destroyAllWindows()

    # anything not completely white (1) is considered occluded (0).
    occ_map = np.floor(occ_map_img)
    # print("raw occupancy grid:\n",occ_map)
    # determine index pairs to select all neighbors when ballooning obstacles.
    nbrs = []
    for i in range(-Config.params["OCC_MAP_BALLOON_AMT"], Config.params["OCC_MAP_BALLOON_AMT"]+1):
        for j in range(-Config.params["OCC_MAP_BALLOON_AMT"], Config.params["OCC_MAP_BALLOON_AMT"]+1):
            nbrs.append((i, j))
    # remove 0,0 which is just the parent cell.
    nbrs.remove((0,0))
    # expand all occluded cells outwards.
    for i in range(len(occ_map)):
        for j in range(len(occ_map[0])):
            if occ_map_img[i][j] != 1: # occluded.
                # mark all neighbors as occluded.
                for chg in nbrs:
                    occ_map[max(0, min(i+chg[0], Config.params["OCC_MAP_SIZE"]-1))][max(0, min(j+chg[1], Config.params["OCC_MAP_SIZE"]-1))] = 0
    # print("inflated map:\n",occ_map)
    occ_map = np.float32(np.array(occ_map))
    # show value distribution in occ_map.
    freqs = [0, 0]
    for i in range(len(occ_map)):
        for j in range(len(occ_map[0])):
            if occ_map[i][j] == 0:
                freqs[0] += 1
            else:
                freqs[1] += 1
    print("Occ map value frequencies: "+str(freqs[1])+" free, "+str(freqs[0])+" occluded.")

    # turn them into Image messages to publish for other nodes.
    bridge = CvBridge()
    occ_map_pub.publish(bridge.cv2_to_imgmsg(occ_map, encoding="passthrough"))
    color_map_pub.publish(bridge.cv2_to_imgmsg(color_map, encoding="passthrough"))


def main():
    global lm_pub, true_map_pub, true_pose_pub, cmd_pub, occ_map_pub, color_map_pub, x_v
    rospy.init_node('sim_node')

    # read command line args.
    if len(sys.argv) < 4:
        rospy.logerr("Required params (occ_map_img, landmark_map, precompute_trajectory) not provided to sim_node.")
        exit()
    else:
        occ_map_img = sys.argv[1]
        landmark_map = sys.argv[2]
        precompute_trajectory = sys.argv[3].lower() == "true"

    # change the starting position if using a special map.
    if occ_map_img == "igvc1.png":
        Config.params["x_0"] = 0.0
        Config.params["y_0"] = -8.5
        Config.params["yaw_0"] = 0.0
    elif occ_map_img == "igvc2.png":
        Config.params["x_0"] = 8.0
        Config.params["y_0"] = 0.0
        Config.params["yaw_0"] = pi/2
    # init vehicle pose from config params.
    x_v = [Config.params["x_0"], Config.params["y_0"], Config.params["yaw_0"]]
    # publish initial pose for EKF.
    init_pose_pub = rospy.Publisher("/truth/init_veh_pose",Vector3, queue_size=1)
    rospy.sleep(1)
    init_pose_pub.publish(Vector3(x=x_v[0], y=x_v[1], z=x_v[2]))

    # subscribe to odom commands.
    rospy.Subscriber("/command", Command, get_cmd, queue_size=1)
    # publish odom commands (either just one to kick things off, or full traj).
    cmd_pub = rospy.Publisher("/command", Command, queue_size=1)

    # publish landmark detections: [id1,r1,b1,...idN,rN,bN]
    lm_pub = rospy.Publisher("/landmark", Float32MultiArray, queue_size=1)

    # publish ground truth for the plotter.
    true_pose_pub = rospy.Publisher("/truth/veh_pose",Vector3, queue_size=1)
    true_map_pub = rospy.Publisher("/truth/landmarks",Float32MultiArray, queue_size=1)
    occ_map_pub = rospy.Publisher("/truth/occ_grid", Image, queue_size=1)
    color_map_pub = rospy.Publisher("/truth/color_map", Image, queue_size=1)

    rospy.sleep(1)
    # initialize the sim.
    init_sim(occ_map_img, landmark_map, precompute_trajectory)

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
