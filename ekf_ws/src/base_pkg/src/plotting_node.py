#!/usr/bin/env python3

"""
Create a live plot of the true and estimated states.
"""

import rospy, rospkg, yaml
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from base_pkg.msg import EKFState, UKFState, PoseGraphState
from matplotlib.backend_bases import MouseButton
from matplotlib import pyplot as plt
import numpy as np
import atexit
from math import cos, sin, pi, atan2, remainder, tau
from cv_bridge import CvBridge

# Suppress constant matplotlib warnings about thread safety.
import warnings
warnings.filterwarnings("ignore")

############ GLOBAL VARIABLES ###################
# store all plots objects we want to be able to remove later.
plots = {"lm_cov_est" : {}}
# output filename prefix.
fname = ""
# publish clicked point on map for planner.
goal_pub = None
# points clicked on the map, if using LIST_CLICKED_POINTS mode.
clicked_points = []
# queue of true poses so we only show the one corresponding to the current estimate.
true_poses = []

# PoseGraphState messages received by plotting_node.
graph_before_optimization = None
graph_after_optimization = None
#################################################


def cov_to_ellipse(P_v):
    """
    Given the state covariance matrix,
    compute points to plot representative ellipse.
    """
    # need to take only first 2x2 for veh position cov.
    cov = P_v[0:2,0:2]
    # get eigenvectors of cov.
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]; vecs = vecs[:,order]
    # fix negative eigenvalues killing the lm ellipses.
    vals = [abs(v) for v in vals]
    # get rotation angle.
    theta = np.arctan2(*vecs[:,0][::-1])
    w, h = config["plotter"]["cov_std_dev"] * 2 * np.sqrt(vals)
    # create parametric ellipse.
    t = np.linspace(0, 2*pi, 100)
    ell = np.array([w*np.cos(t) , h*np.sin(t)])
    # compute rotation transform.
    R_ell = np.array([[cos(theta), -sin(theta)],[sin(theta) , cos(theta)]])
    # apply rotation to ellipse.
    ell_rot = np.zeros((2,ell.shape[1]))
    for i in range(ell.shape[1]):
        ell_rot[:,i] = np.dot(R_ell,ell[:,i])
    return ell_rot


def update_plot(filter:str, msg):
    # draw everything for this timestep.
    global plots

    # if we detect that pose-graph-slam has finished, kill the current plot and switch to that one.
    if graph_before_optimization is not None and graph_after_optimization is not None:
        # kill the current plot.
        rospy.logwarn("PLT: killing current plot.")
        plt.clf()
        # start the pose-graph plot.
        rospy.logwarn("PLT: starting PGS-Viz loop.")

        """
        Plot the graph encoded in a PoseGraphState message.
        """
        # plot all landmark position estimates.
        if len(graph_before_optimization.landmarks) > 0: # there might be none.
            lm_x = [graph_before_optimization.landmarks[i] for i in range(0,len(graph_before_optimization.landmarks),2)]
            lm_y = [graph_before_optimization.landmarks[i] for i in range(1,len(graph_before_optimization.landmarks),2)]
            plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black", zorder=3)

        # plot all vehicle poses.
        for i in range(graph_before_optimization.num_iterations):
            # draw a single pt with arrow to represent each veh pose.
            plt.arrow(graph_before_optimization.x_v[i], graph_before_optimization.y_v[i], config["plotter"]["arrow_len"]*cos(graph_before_optimization.yaw_v[i]), config["plotter"]["arrow_len"]*sin(graph_before_optimization.yaw_v[i]), facecolor="blue", width=0.1, zorder=4, edgecolor="black")

        # plot all connections.
        # we know a connection exists between every vehicle pose and the pose on the immediate previous/next iterations.
        plt.plot(graph_before_optimization.x_v, graph_before_optimization.y_v, color="blue", zorder=0)
        # measurement connections are not fully-connected, but rather encoded in the message.
        if len(graph_before_optimization.meas_connections) > 0: # there could possibly be none.
            for j in range(len(graph_before_optimization.meas_connections) // 2):
                iter_veh_pose = graph_before_optimization.meas_connections[2*j]
                landmark_index = graph_before_optimization.meas_connections[2*j+1]
                # plot a line between the specified vehicle pose and landmark.
                plt.plot([graph_before_optimization.x_v[iter_veh_pose], lm_x[landmark_index]], [graph_before_optimization.x_y[iter_veh_pose], lm_y[landmark_index]], color="red", zorder=0)

        # do the plotting.
        plt.draw()
        plt.pause(0.00000000001)

        # exit this node when PGS viz is done.
        rospy.sleep(5)
        rospy.logwarn("PLT: skipping plot update loop.")
        return
    
    ###################### TIMESTEP #####################
    if "timestep" in plots.keys():
        plots["timestep"].remove()
    plots["timestep"] = plt.text(-config["map"]["bound"], config["map"]["bound"], 't = '+str(msg.timestep), horizontalalignment='left', verticalalignment='bottom', zorder=2)
    #################### TRUE POSE #########################
    if config["plotter"]["show_true_traj"] and msg.timestep <= len(true_poses):
        pose = true_poses[msg.timestep-1]
        # plot the current veh pos, & remove previous.
        if "veh_pos_true" in plots.keys():
            plots["veh_pos_true"].remove()
        plots["veh_pos_true"] = plt.arrow(pose.x, pose.y, config["plotter"]["arrow_len"]*cos(pose.z), config["plotter"]["arrow_len"]*sin(pose.z), color="blue", width=0.1, zorder=2)

    ####################### EKF/UKF SLAM #########################
    if filter in ["ekf", "ukf"]:
        plt.title(filter.upper()+"-Estimated Trajectory and Landmarks")
        # compute length of state to use throughout. n = 3+2M
        n = int(len(msg.P)**(1/2))
        ################ VEH POS #################
        # plot current estimated veh pos.
        if not config["plotter"]["show_entire_traj"] and "veh_pos_est" in plots.keys():
            plots["veh_pos_est"].remove()
            del plots["veh_pos_est"]
        # draw a single pt with arrow to represent current veh pose.
        plots["veh_pos_est"] = plt.arrow(msg.x_v, msg.y_v, config["plotter"]["arrow_len"]*cos(msg.yaw_v), config["plotter"]["arrow_len"]*sin(msg.yaw_v), facecolor="green", width=0.1, zorder=4, edgecolor="black")

        ################ VEH COV ##################
        if config["plotter"]["show_veh_ellipse"]:
            # compute parametric ellipse for veh covariance.
            veh_ell = cov_to_ellipse(np.array([[msg.P[0],msg.P[1]], [msg.P[n],msg.P[n+1]]]))
            # remove old ellipses.
            if not config["plotter"]["show_entire_traj"] and "veh_cov_est" in plots.keys():
                plots["veh_cov_est"].remove()
                del plots["veh_cov_est"]
            # plot the ellipse.
            plots["veh_cov_est"], = plt.plot(msg.x_v+veh_ell[0,:] , msg.y_v+veh_ell[1,:],'lightgrey', zorder=1)

        ############## LANDMARK EST ##################
        lm_x = [msg.landmarks[i] for i in range(1,len(msg.landmarks),3)]
        lm_y = [msg.landmarks[i] for i in range(2,len(msg.landmarks),3)]
        # remove old landmark estimates.
        if "lm_pos_est" in plots.keys():
            plots["lm_pos_est"].remove()
            del plots["lm_pos_est"]
        # plot new landmark estimates.
        plots["lm_pos_est"] = plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black", zorder=3)

        ############## LANDMARK COV ###################
        if config["plotter"]["show_landmark_ellipses"]:
            # plot new landmark covariances.
            for i in range(len(msg.landmarks) // 3):
                # replace previous if it's been plotted before.
                lm_id = msg.landmarks[i*3] 
                if lm_id in plots["lm_cov_est"].keys():
                    plots["lm_cov_est"][lm_id].remove()
                    del plots["lm_cov_est"][lm_id]
                # extract 2x2 cov for this landmark.
                lm_ell = cov_to_ellipse(np.array([[msg.P[3+2*i],msg.P[4+2*i]],[msg.P[n+3+2*i],msg.P[n+4+2*i]]]))
                # plot its ellipse.
                plots["lm_cov_est"][lm_id], = plt.plot(lm_x[i]+lm_ell[0,:], lm_y[i]+lm_ell[1,:], 'orange', zorder=1)
        
        ############## UKF SIGMA POINTS ##################
        if filter == "ukf":
            # extract length of vectors in sigma pts (not necessarily = n).
            n_sig = int(((1+8*len(msg.X))**(1/2) - 1) / 4) # soln to n*(2n+1)=len.
            # check if we're using UKF with 3x1 or 4x1 veh state.
            veh_len = 3 if n_sig % 2 == 1 else 4
            ############## VEH POSE SIGMA POINTS #################
            # only show sigma points' veh pose, not landmark info.
            if config["plotter"]["plot_ukf_arrows"]:
                # plot as arrows (slow).
                if "veh_sigma_pts" in plots.keys() and len(plots["veh_sigma_pts"].keys()) > 0:
                    for i in plots["veh_sigma_pts"].keys():
                        plots["veh_sigma_pts"][i].remove()
                plots["veh_sigma_pts"] = {}
                # draw a pt with arrow for all sigma pts.
                for i in range(0, 2*n_sig+1):
                    yaw = msg.X[i*n_sig+2] if veh_len == 3 else remainder(atan2(msg.X[i*n_sig+3], msg.X[i*n_sig+2]), tau)
                    plots["veh_sigma_pts"][i] = plt.arrow(msg.X[i*n_sig], msg.X[i*n_sig+1], config["plotter"]["arrow_len"]*cos(yaw), config["plotter"]["arrow_len"]*sin(yaw), color="cyan", width=0.1)
            else: # just show x,y of pts.
                X_x = [msg.X[i*n_sig] for i in range(0,2*n_sig+1)]
                X_y = [msg.X[i*n_sig+1] for i in range(0,2*n_sig+1)]
                # remove old points.
                if "veh_sigma_pts" in plots.keys():
                    plots["veh_sigma_pts"].remove()
                    del plots["veh_sigma_pts"]
                # plot sigma points.
                plots["veh_sigma_pts"] = plt.scatter(X_x, X_y, s=30, color="tab:cyan", zorder=2)

            ################# LANDMARK SIGMA POINTS ##################
            if config["plotter"]["show_landmark_sigma_pts"]:
                # plot all landmark sigma pts.
                X_lm_x = []; X_lm_y = []
                for j in range(2*n_sig+1):
                    X_lm_x += [msg.X[j*n_sig+i] for i in range(veh_len,n_sig,2)]
                    X_lm_y += [msg.X[j*n_sig+i+1] for i in range(veh_len,n_sig,2)]
                # remove old points.
                if "lm_sigma_pts" in plots.keys():
                    plots["lm_sigma_pts"].remove()
                    del plots["lm_sigma_pts"]
                # plot sigma points.
                plots["lm_sigma_pts"] = plt.scatter(X_lm_x, X_lm_y, s=30, color="tab:cyan", zorder=1)

    # force desired window region.
    plt.xlim(display_region)
    plt.ylim(display_region)
    # do the plotting.
    plt.draw()
    plt.pause(0.00000000001)


# get the state published by the EKF.
def get_ekf_state(msg):
    global fname
    # set filename for EKF.
    fname = "ekf"
    # update the plot.
    update_plot("ekf", msg)

# get the state published by the UKF.
def get_ukf_state(msg):
    global fname
    # set filename for UKF.
    fname = "ukf"
    # update the plot.
    update_plot("ukf", msg)

# get the factor graphs published by Pose Graph SLAM filter.
def get_pose_graph_initial(msg):
    # set this graph for our one-time plotter.
    global graph_before_optimization
    graph_before_optimization = msg
    rospy.loginfo("PLT: plotting node got PGS initial.")

def get_pose_graph_result(msg):
    # set this graph for our one-time plotter.
    global graph_after_optimization
    graph_after_optimization = msg
    rospy.loginfo("PLT: plotting node got PGS result.")

def save_plot(pkg_path):
    # save plot we've been building upon exit.
    # save to a file in pkg/plots directory.
    if fname != "" and config["plotter"]["save_final_map"]:
        plt.savefig(pkg_path+"/plots/"+fname+"_demo.png", format='png')

def get_true_pose(msg):
    if not config["plotter"]["show_true_traj"]: return
    # save the messages to a queue so they can be shown with the corresponding estimate.
    global true_poses
    true_poses.append(msg)

def get_true_landmark_map(msg):
    if not config["plotter"]["show_true_landmark_map"]: return
    # rospy.loginfo("Ground truth map received by plotting node.")
    lm_x = [msg.data[i] for i in range(1,len(msg.data),3)]
    lm_y = [msg.data[i] for i in range(2,len(msg.data),3)]
    # plot the true landmark positions to compare to estimates.
    plt.scatter(lm_x, lm_y, s=30, color="white", edgecolors="black", zorder=2)

def on_click(event):
    # global clicked_points
    if event.button is MouseButton.RIGHT:
        # kill the node.
        rospy.loginfo("Killing plotting_node on right click.")
        exit()
    elif event.button is MouseButton.LEFT:
        if config["plotter"]["list_clicked_points"]:
            clicked_points.append((event.xdata, event.ydata))
            print(clicked_points)
        # publish new goal pt for the planner.
        goal_pub.publish(Vector3(x=event.xdata, y=event.ydata))
        # show this point so it appears even when not using A* planning.
        get_planned_path(Float32MultiArray(data=[event.xdata, event.ydata]))

def get_color_map(msg):
    if not config["plotter"]["show_occ_map"]: return
    # get the true occupancy grid map image.
    bridge = CvBridge()
    color_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # add the true map image to the plot. extent=(L,R,B,T) gives display bounds.
    edge = config["map"]["bound"]
    plt.imshow(color_map, zorder=0, extent=[-edge, edge, -edge, edge])


def get_planned_path(msg):
    global plots
    # get the set of points that A* determined to get to the clicked goal.
    # remove and update the path if we've drawn it already.
    if "planned_path" in plots.keys():
        plots["planned_path"].remove()
        del plots["planned_path"]
    if "goal_pt" in plots.keys():
            plots["goal_pt"].remove()
            del plots["goal_pt"]
    # only draw if the path is non-empty.
    if len(msg.data) > 1:
        plots["planned_path"] = plt.scatter([msg.data[i] for i in range(0,len(msg.data),2)], [msg.data[i] for i in range(1,len(msg.data),2)], s=12, color="purple", zorder=1)
        # show the goal point (end of path).
        plots["goal_pt"] = plt.scatter(msg.data[-2], msg.data[-1], color="yellow", edgecolors="black", s=40, zorder=2)

def main():
    global goal_pub
    rospy.init_node('plotting_node')

    # read configs.
    # find the filepath to the params file.
    rospack = rospkg.RosPack()
    global base_pkg_path, config
    base_pkg_path = rospack.get_path('base_pkg')
    # open the yaml and read all params.
    with open(base_pkg_path+'/config/params.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # compute any additional needed global configs.
    global config_shift, config_scale, display_region
    # occ map <-> lm coords transform params.
    config_shift = config["map"]["occ_map_size"] / 2
    config_scale = config["map"]["bound"] / config_shift
    # size of region plotter will display.
    display_region = [config["map"]["bound"] * config["plotter"]["display_region_mult"] * sign for sign in (-1, 1)]

    # make live plot bigger.
    plt.rcParams["figure.figsize"] = (9,9)

    # when the node exits, make the plot.
    atexit.register(save_plot, base_pkg_path)

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    rospy.Subscriber("/state/ukf", UKFState, get_ukf_state, queue_size=1)
    rospy.Subscriber("/state/pose_graph/initial", PoseGraphState, get_pose_graph_initial, queue_size=1)
    rospy.Subscriber("/state/pose_graph/result", PoseGraphState, get_pose_graph_result, queue_size=1)
    # subscribe to ground truth.
    rospy.Subscriber("/truth/veh_pose", Vector3, get_true_pose, queue_size=1)
    rospy.Subscriber("/truth/landmarks", Float32MultiArray, get_true_landmark_map, queue_size=1)
    # subscribe to the color map.
    rospy.Subscriber("/truth/color_map", Image, get_color_map, queue_size=1)

    # publish the chosen goal point for the planner.
    goal_pub = rospy.Publisher("/plan/goal", Vector3, queue_size=1)
    # subscribe to planned path to the goal.
    rospy.Subscriber("/plan/path", Float32MultiArray, get_planned_path, queue_size=1)


    # startup the plot.
    plt.figure()
    # set constant plot params.
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    # allow user to click on the plot to set the goal position.
    plt.connect('button_press_event', on_click)
    plt.show()

    

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


# def move_figure(f, x, y):
# # change position that the plot appears.
# # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
#     """Move figure's upper left corner to pixel (x, y)"""
#     backend = matplotlib.get_backend()
#     if backend == 'TkAgg':
#         f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
#     elif backend == 'WXAgg':
#         f.canvas.manager.window.SetPosition((x, y))
#     else:
#         # This works for QT and GTK
#         # You can also use window.setGeometry
#         f.canvas.manager.window.move(x, y)

# binding_id = plt.connect('motion_notify_event', on_move)
# def on_move(event):
#     # get the x and y pixel coords
#     x, y = event.x, event.y
#     if event.inaxes:
#         ax = event.inaxes  # the axes instance
#         print('data coords %f %f' % (event.xdata, event.ydata))
