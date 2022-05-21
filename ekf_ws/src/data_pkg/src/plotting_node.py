#!/usr/bin/env python3

"""
Create a live plot of the true and estimated states.
"""

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from ekf_pkg.msg import EKFState
from pf_pkg.msg import PFState
from matplotlib import pyplot as plt
import numpy as np
import atexit
from math import cos, sin, pi

############ GLOBAL VARIABLES ###################
params = {}
# store all plots objects we want to be able to remove later.
plots = {"lm_cov_est" : {}}
# ground truth.
true_pose = None # current true pose as Vector3.
true_map = None # true landmark map.
# position to display the timestep number.
pos_time_display = None
# output filename prefix.
fname = ""
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
    w, h = params["COV_STD_DEV"] * 2 * np.sqrt(vals)
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
    global plots, pos_time_display, true_map, true_traj
    
    #################### TRUE MAP #######################
    if true_map is not None:
        plt.scatter(true_map[0], true_map[1], s=30, color="white", edgecolors="black")
        # make sure the time text will be on screen but not blocking a lm.
        pos_time_display = (min(true_map[0]), max(true_map[1])+2)
        # this should only run once to avoid wasting time.
        true_map = None

    ################ TRUE TRAJECTORY #####################
    if params["SHOW_TRUE_TRAJ"] and true_pose is not None:
        # plot only the current veh pos.
        if "veh_pos_true" in plots.keys():
            plots["veh_pos_true"].remove()
        plots["veh_pos_true"] = plt.arrow(true_pose.x, true_pose.y, params["ARROW_LEN"]*cos(true_pose.z), params["ARROW_LEN"]*sin(true_pose.z), color="blue", width=0.1)

    ###################### TIMESTEP #####################
    if "timestep" in plots.keys():
        plots["timestep"].remove()
    if pos_time_display is not None:
        plots["timestep"] = plt.text(pos_time_display[0], pos_time_display[1], 't = '+str(msg.timestep), horizontalalignment='center', verticalalignment='bottom')

    ####################### EKF SLAM #########################
    if filter == "ekf":
        ################ VEH POS #################
        # plot current EKF-estimated veh pos.
        if not params["SHOW_ENTIRE_TRAJ"] and "veh_pos_est" in plots.keys():
            plots["veh_pos_est"].remove()
        # draw a single pt with arrow to represent current veh pose.
        plots["veh_pos_est"] = plt.arrow(msg.x_v, msg.y_v, params["ARROW_LEN"]*cos(msg.yaw_v), params["ARROW_LEN"]*sin(msg.yaw_v), color="green", width=0.1)

        ################ VEH COV ##################
        n = int(len(msg.P)**(1/2)) # length of state n = 3+2M
        # compute parametric ellipse for veh covariance.
        veh_ell = cov_to_ellipse(np.array([[msg.P[0],msg.P[1]], [msg.P[n],msg.P[n+1]]]))
        # remove old ellipses.
        if not params["SHOW_ENTIRE_TRAJ"] and "veh_cov_est" in plots.keys():
            plots["veh_cov_est"].remove()
        # plot the ellipse.
        plots["veh_cov_est"], = plt.plot(msg.x_v+veh_ell[0,:] , msg.y_v+veh_ell[1,:],'lightgrey')

        ############## LANDMARK EST ##################
        lm_x = [msg.landmarks[i] for i in range(1,len(msg.landmarks),3)]
        lm_y = [msg.landmarks[i] for i in range(2,len(msg.landmarks),3)]
        # remove old landmark estimates.
        if "lm_pos_est" in plots.keys():
            plots["lm_pos_est"].remove()
        # plot new landmark estimates.
        plots["lm_pos_est"] = plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black")

        ############## LANDMARK COV ###################
        # plot new landmark covariances.
        for i in range(len(msg.landmarks) // 3):
            # replace previous if it's been plotted before.
            lm_id = msg.landmarks[i*3] 
            if lm_id in plots["lm_cov_est"].keys():
                plots["lm_cov_est"][lm_id].remove()
            # extract 2x2 cov for this landmark.
            lm_ell = cov_to_ellipse(np.array([[msg.P[3+2*i],msg.P[4+2*i]],[msg.P[n+3+2*i],msg.P[n+4+2*i]]]))
            # plot its ellipse.
            plots["lm_cov_est"][lm_id], = plt.plot(lm_x[i]+lm_ell[0,:] , lm_y[i]+lm_ell[1,:],'orange')

    ############## PARTICLE FILTER LOCALIZATION #####################
    elif filter == "pf":
        ########## PARTICLE SET ###############
        if params["PLOT_PARTICLE_ARROWS"]:
            # plot as arrows (slow).
            if "particle_set" in plots.keys() and len(plots["particle_set"].keys()) > 0:
                for i in plots["particle_set"].keys():
                    plots["particle_set"][i].remove()
            plots["particle_set"] = {}
            # draw a pt with arrow for all particles.
            for i in range(len(msg.x)):
                plots["particle_set"][i] = plt.arrow(msg.x[i], msg.y[i], params["ARROW_LEN"]*cos(msg.yaw[i]), params["ARROW_LEN"]*sin(msg.yaw[i]), color="red", width=0.1)
        else:
            # plot as points (faster).
            if "particle_set" in plots.keys():
                plots["particle_set"].remove()
            # draw entire particle set at once.
            plots["particle_set"] = plt.scatter([msg.x[i] for i in range(len(msg.x))], [msg.y[i] for i in range(len(msg.y))], s=8, color="red")

    # do the plotting.
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-1.5*params["MAP_BOUND"],1.5*params["MAP_BOUND"]])
    plt.ylim([-1.5*params["MAP_BOUND"],1.5*params["MAP_BOUND"]])
    plt.title("EKF-Estimated Trajectory and Landmarks")
    plt.draw()
    plt.pause(0.00000000001)


# get the state published by the EKF.
def get_ekf_state(msg):
    global fname
    # set filename for EKF.
    fname = "ekf"
    # update the plot.
    update_plot("ekf", msg)

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
    if fname != "":
        plt.savefig(pkg_path+"/plots/"+fname+"_demo.png", format='png')

def get_true_pose(msg):
    # save the true pose to plot it along w cur state.
    global true_pose
    true_pose = msg

def get_true_map(msg):
    rospy.loginfo("Ground truth map received by plotting node.")
    # plot the true map to compare to estimates.
    global true_map
    lm_x = [msg.data[i] for i in range(1,len(msg.data),3)]
    lm_y = [msg.data[i] for i in range(2,len(msg.data),3)]
    true_map = [lm_x, lm_y]
    

def main():
    rospy.init_node('plotting_node')

    # make live plot bigger.
    plt.rcParams["figure.figsize"] = (9,9)

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')
    # read params.
    read_params(pkg_path)

    # when the node exits, make the plot.
    atexit.register(save_plot, pkg_path)

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    rospy.Subscriber("/state/pf", PFState, get_pf_state, queue_size=1)

    # subscribe to ground truth.
    rospy.Subscriber("/truth/veh_pose",Vector3, get_true_pose, queue_size=1)
    rospy.Subscriber("/truth/landmarks",Float32MultiArray, get_true_map, queue_size=1)

    # startup the plot.
    plt.ion()
    plt.show()

    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass