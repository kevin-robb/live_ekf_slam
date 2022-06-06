#!/usr/bin/env python3

"""
Create a live plot of the true and estimated states.
"""

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from data_pkg.msg import EKFState, UKFState, PFState
from matplotlib.backend_bases import MouseButton
from matplotlib import pyplot as plt
import numpy as np
import atexit
from math import cos, sin, pi
from cv_bridge import CvBridge
from import_params import Config

############ GLOBAL VARIABLES ###################
# store all plots objects we want to be able to remove later.
plots = {"lm_cov_est" : {}}
# output filename prefix.
fname = ""
# publish clicked point on map for planner.
goal_pub = None
# points clicked on the map, if using LIST_CLICKED_POINTS mode.
clicked_points = []
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
    w, h = Config.params["COV_STD_DEV"] * 2 * np.sqrt(vals)
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
    
    ###################### TIMESTEP #####################
    if "timestep" in plots.keys():
        plots["timestep"].remove()
    plots["timestep"] = plt.text(-Config.params["MAP_BOUND"], Config.params["MAP_BOUND"], 't = '+str(msg.timestep), horizontalalignment='left', verticalalignment='bottom', zorder=2)

    ####################### EKF/UKF SLAM #########################
    if filter in ["ekf", "ukf"]:
        plt.title(filter.upper()+"-Estimated Trajectory and Landmarks")
        # compute length of state to use throughout. n = 3+2M
        n = int(len(msg.P)**(1/2))
        ################ VEH POS #################
        # plot current estimated veh pos.
        if not Config.params["SHOW_ENTIRE_TRAJ"] and "veh_pos_est" in plots.keys():
            plots["veh_pos_est"].remove()
            del plots["veh_pos_est"]
        # draw a single pt with arrow to represent current veh pose.
        plots["veh_pos_est"] = plt.arrow(msg.x_v, msg.y_v, Config.params["ARROW_LEN"]*cos(msg.yaw_v), Config.params["ARROW_LEN"]*sin(msg.yaw_v), facecolor="green", width=0.1, zorder=2, edgecolor="black")

        ################ VEH COV ##################
        if Config.params["SHOW_VEH_ELLIPSE"]:
            # compute parametric ellipse for veh covariance.
            veh_ell = cov_to_ellipse(np.array([[msg.P[0],msg.P[1]], [msg.P[n],msg.P[n+1]]]))
            # remove old ellipses.
            if not Config.params["SHOW_ENTIRE_TRAJ"] and "veh_cov_est" in plots.keys():
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
        plots["lm_pos_est"] = plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black", zorder=1)

        ############## LANDMARK COV ###################
        if Config.params["SHOW_LM_ELLIPSES"]:
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
                plots["lm_cov_est"][lm_id], = plt.plot(lm_x[i]+lm_ell[0,:] , lm_y[i]+lm_ell[1,:],'orange', zorder=1)
        
        ############## UKF SIGMA POINTS ##################
        if filter == "ukf":
            # TODO make numpy matrix for X instead of dealing with single array.
            if Config.params["PLOT_UKF_ARROWS"]:
                # plot as arrows (slow).
                if "sigma_pts" in plots.keys() and len(plots["sigma_pts"].keys()) > 0:
                    for i in plots["sigma_pts"].keys():
                        plots["sigma_pts"][i].remove()
                plots["sigma_pts"] = {}
                # draw a pt with arrow for all sigma pts.
                rospy.logwarn(str(msg.X))
                for i in range(0, len(msg.X), n):
                    plots["sigma_pts"][i] = plt.arrow(msg.X[i], msg.X[i+1], Config.params["ARROW_LEN"]*cos(msg.X[i+2]), Config.params["ARROW_LEN"]*sin(msg.X[i+2]), color="cyan", width=0.1)
            else: # just show x,y of pts.
                X_x = [msg.X[i] for i in range(0,len(msg.X),3)]
                X_y = [msg.X[i] for i in range(1,len(msg.X),3)]
                # remove old points.
                if "sigma_pts" in plots.keys():
                    plots["sigma_pts"].remove()
                    del plots["sigma_pts"]
                # plot sigma points.
                plots["sigma_pts"] = plt.scatter(X_x, X_y, s=30, color="tab:cyan", zorder=1)

    ############## PARTICLE FILTER LOCALIZATION #####################
    elif filter == "pf":
        plt.title("PF-Estimated Vehicle Pose")
        ########## PARTICLE SET ###############
        if Config.params["PLOT_PF_ARROWS"]:
            # plot as arrows (slow).
            if "particle_set" in plots.keys() and len(plots["particle_set"].keys()) > 0:
                for i in plots["particle_set"].keys():
                    plots["particle_set"][i].remove()
            plots["particle_set"] = {}
            # draw a pt with arrow for all particles.
            for i in range(len(msg.x)):
                plots["particle_set"][i] = plt.arrow(msg.x[i], msg.y[i], Config.params["ARROW_LEN"]*cos(msg.yaw[i]), Config.params["ARROW_LEN"]*sin(msg.yaw[i]), color="red", width=0.1)
        else:
            # plot as points (faster).
            if "particle_set" in plots.keys():
                plots["particle_set"].remove()
            # draw entire particle set at once.
            plots["particle_set"] = plt.scatter([msg.x[i] for i in range(len(msg.x))], [msg.y[i] for i in range(len(msg.y))], s=8, color="red")

    # force desired window region.
    plt.xlim(Config.params["DISPLAY_REGION"])
    plt.ylim(Config.params["DISPLAY_REGION"])
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

# get the state published by the PF.
def get_pf_state(msg):
    global fname
    # set filename for PF output.
    fname = "pf"
    # update the plot.
    update_plot("pf", msg)

def save_plot(pkg_path):
    # save plot we've been building upon exit.
    # save to a file in pkg/plots directory.
    if fname != "" and Config.params["SAVE_FINAL_MAP"]:
        plt.savefig(pkg_path+"/plots/"+fname+"_demo.png", format='png')

def get_true_pose(msg):
    if not Config.params["SHOW_TRUE_TRAJ"]: return
    global plots
    # plot the current veh pos, & remove previous.
    if "veh_pos_true" in plots.keys():
        plots["veh_pos_true"].remove()
    plots["veh_pos_true"] = plt.arrow(msg.x, msg.y, Config.params["ARROW_LEN"]*cos(msg.z), Config.params["ARROW_LEN"]*sin(msg.z), color="blue", width=0.1, zorder=1)

def get_true_landmark_map(msg):
    if not Config.params["SHOW_TRUE_LM_MAP"]: return
    # rospy.loginfo("Ground truth map received by plotting node.")
    lm_x = [msg.data[i] for i in range(1,len(msg.data),3)]
    lm_y = [msg.data[i] for i in range(2,len(msg.data),3)]
    # plot the true landmark positions to compare to estimates.
    plt.scatter(lm_x, lm_y, s=30, color="white", edgecolors="black", zorder=1)

def on_click(event):
    # global clicked_points
    if event.button is MouseButton.RIGHT:
        # kill the node.
        rospy.loginfo("Killing plotting_node on right click.")
        exit()
    elif event.button is MouseButton.LEFT:
        if Config.params["LIST_CLICKED_POINTS"]:
            clicked_points.append((event.xdata, event.ydata))
            print(clicked_points)
        # publish new goal pt for the planner.
        goal_pub.publish(Vector3(x=event.xdata, y=event.ydata))
        # show this point so it appears even when not using A* planning.
        get_planned_path(Float32MultiArray(data=[event.xdata, event.ydata]))

def get_color_map(msg):
    if not Config.params["SHOW_OCC_MAP"]: return
    # get the true occupancy grid map image.
    bridge = CvBridge()
    color_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # add the true map image to the plot. extent=(L,R,B,T) gives display bounds.
    edge = Config.params["MAP_BOUND"]
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

    # make live plot bigger.
    plt.rcParams["figure.figsize"] = (9,9)

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')

    # when the node exits, make the plot.
    atexit.register(save_plot, pkg_path)

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    rospy.Subscriber("/state/ukf", UKFState, get_ukf_state, queue_size=1)
    rospy.Subscriber("/state/pf", PFState, get_pf_state, queue_size=1)
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
