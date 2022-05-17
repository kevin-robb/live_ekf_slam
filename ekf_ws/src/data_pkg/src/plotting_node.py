#!/usr/bin/env python3

import rospy
import rospkg
from std_msgs.msg import Float32MultiArray
from ekf_pkg.msg import EKFState
from matplotlib import pyplot as plt
import numpy as np
import atexit
from math import cos, sin, pi

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

############ GLOBAL VARIABLES ###################
# map bounds.
MAP_BOUND = 10
# state data from the EKF to use for plotting.
# position/hdg for all times.
veh_x = []; veh_y = []; veh_th = []
# current estimate for all landmark positions.
lm_x = []; lm_y = []
# ground truth veh pose.
true_pose = None; true_map = None
# plotting stuff.
SHOW_ENTIRE_TRAJ = False
SHOW_TRUE_TRAJ = True
ARROW_LEN = 0.1 # length of pose arrow.
NUM_STD_DEV = 1 # number of std dev to include in cov ellipse.
timestep_num = 1; time_text = None; time_text_position = None
lm_pts = None; veh_pts = None; ell_pts = None; veh_true = None
lm_cov = {}
# set of PF particle plots.
particle_plots = None; true_pt = None
# output filename prefix.
fname = ""
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
    w, h = NUM_STD_DEV * 2 * np.sqrt(vals)
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


# get the state published by the EKF.
def get_ekf_state(msg):
    global veh_x, veh_y, veh_th, lm_x, lm_y, lm_pts, veh_pts, ell_pts, veh_true, true_map, true_pose, timestep_num, time_text, time_text_position, fname, lm_cov
    # set filename for EKF.
    fname = "ekf"
    # save what we want to plot.
    # extract the covariance matrix for the vehicle.
    n = int(len(msg.P)**(1/2))
    P_v = np.array([[msg.P[0],msg.P[1],msg.P[2]],
                    [msg.P[n],msg.P[n+1],msg.P[n+2]],
                    [msg.P[2*n],msg.P[2*n+1],msg.P[2*n+2]]])
    # rest is the state.
    veh_x.append(msg.x_v)
    veh_y.append(msg.y_v)
    veh_th.append(msg.yaw_v)
    if n > 3:
        lm_x = [msg.landmarks[i] for i in range(1,len(msg.landmarks),3)]
        lm_y = [msg.landmarks[i] for i in range(2,len(msg.landmarks),3)]

    # plot ground truth map.
    if true_map is not None:
        plt.scatter(true_map[0], true_map[1], s=30, color="white", edgecolors="black")
        # make sure the time text will be on screen but not blocking a lm.
        time_text_position = (min(true_map[0]), max(true_map[1])+1)
        # this should only run once to avoid wasting time.
        true_map = None

    # plot full ground truth trajectory.
    if SHOW_TRUE_TRAJ and true_pose is not None:
        for timestep in range(0,len(true_pose)//3):
            plt.arrow(true_pose[timestep*3], true_pose[timestep*3+1], ARROW_LEN*cos(true_pose[timestep*3+2]), ARROW_LEN*sin(true_pose[timestep*3+2]), color="blue", width=0.01)
        true_pose = None

    # compute parametric ellipse for veh covariance.
    ell_rot = cov_to_ellipse(P_v)
    # remove old ellipses.
    if not SHOW_ENTIRE_TRAJ and ell_pts is not None:
        ell_pts.remove()
    # plot the ellipse.
    ell_pts, = plt.plot(msg.x_v+ell_rot[0,:] , msg.y_v+ell_rot[1,:],'lightgrey')

    # remove old landmark estimates and covariances.
    if lm_pts is not None:
        lm_pts.remove()
    # plot new landmark estimates.
    lm_pts = plt.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black")
    # plot new landmark covariances.
    for i in range(len(msg.landmarks) // 3):
        # remove if it's been plotted before.
        lm_id = msg.landmarks[i*3] 
        if lm_id in lm_cov.keys() and lm_cov[lm_id] is not None:
            lm_cov[lm_id].remove()
            lm_cov[lm_id] = None
        # extract 2x2 cov for this landmark.
        lm_ell = cov_to_ellipse(np.array([[msg.P[3+2*i],msg.P[4+2*i]],[msg.P[n+3+2*i],msg.P[n+4+2*i]]]))
        # plot its ellipse.
        lm_cov[lm_id], = plt.plot(lm_x[i]+lm_ell[0,:] , lm_y[i]+lm_ell[1,:],'orange')

    # plot current EKF-estimated veh pos.
    if not SHOW_ENTIRE_TRAJ and veh_pts is not None:
        veh_pts.remove()
    # draw a single pt with arrow to represent current veh pose.
    veh_pts = plt.arrow(msg.x_v, msg.y_v, ARROW_LEN*cos(msg.yaw_v), ARROW_LEN*sin(msg.yaw_v), color="green", width=0.1)

    # show the timestep on the plot.
    if time_text is not None:
        time_text.remove()
    if time_text_position is not None:
        time_text = plt.text(time_text_position[0], time_text_position[1], 't = '+str(timestep_num), horizontalalignment='center', verticalalignment='bottom')
    timestep_num += 1

    # do the plotting.
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-1.5*MAP_BOUND,1.5*MAP_BOUND])
    plt.ylim([-1.5*MAP_BOUND,1.5*MAP_BOUND])
    plt.title("EKF-Estimated Trajectory and Landmarks")
    plt.draw()
    plt.pause(0.00000000001)


# get the state published by the PF.
def get_pf_state(msg):
    global true_map, true_pose, timestep_num, time_text, time_text_position, particle_plots, fname, true_pt
    # set filename for PF output.
    fname = "pf"
    # plot ground truth map.
    if true_map is not None:
        plt.scatter(true_map[0], true_map[1], s=30, color="white", edgecolors="black")
        # make sure the time text will be on screen but not blocking a lm.
        time_text_position = (min(true_map[0]), max(true_map[1])+1)
        # this should only run once to avoid wasting time.
        true_map = None

    # # plot full ground truth trajectory.
    # if SHOW_TRUE_TRAJ and true_pose is not None:
    #     for timestep in range(0,len(true_pose)//3):
    #         plt.arrow(true_pose[timestep*3], true_pose[timestep*3+1], ARROW_LEN*cos(true_pose[timestep*3+2]), ARROW_LEN*sin(true_pose[timestep*3+2]), color="blue", width=0.01)
    #     true_pose = None

    # plot particle set.
    PLOT_PARTICLE_ARROWS = False
    if PLOT_PARTICLE_ARROWS:
        # plot as arrows.
        if particle_plots is not None and len(particle_plots) > 0:
            for i in range(len(particle_plots)):
                particle_plots[i].remove()
        particle_plots = []
        # draw a pt with arrow for all particles.
        for i in range(len(msg.data) // 3):
            pp = plt.arrow(msg.data[i*3], msg.data[i*3+1], ARROW_LEN*cos(msg.data[i*3+2]), ARROW_LEN*sin(msg.data[i*3+2]), color="red", width=0.1)
            particle_plots.append(pp)
    else:
        # plot as points (faster).
        if particle_plots is not None:
            particle_plots.remove()
        # draw entire particle set at once.
        particle_plots = plt.scatter([msg.data[i] for i in range(0,len(msg.data),3)], [msg.data[i] for i in range(1,len(msg.data),3)], s=12, color="red")

    # show the current true pose.
    if SHOW_TRUE_TRAJ and true_pose is not None:
        # remove previous.
        if true_pt is not None:
            true_pt.remove()
        # draw current.
        true_pt = plt.arrow(true_pose[timestep_num*3], true_pose[timestep_num*3+1], ARROW_LEN*cos(true_pose[timestep_num*3+2]), ARROW_LEN*sin(true_pose[timestep_num*3+2]), color="blue", width=0.1)

    # show the timestep on the plot.
    if time_text is not None:
        time_text.remove()
    if time_text_position is not None:
        time_text = plt.text(time_text_position[0], time_text_position[1], 't = '+str(timestep_num), horizontalalignment='center', verticalalignment='bottom')
    timestep_num += 1

    # do the plotting.
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim([-1.5*MAP_BOUND,1.5*MAP_BOUND])
    plt.ylim([-1.5*MAP_BOUND,1.5*MAP_BOUND])
    plt.title("EKF-Estimated Trajectory and Landmarks")
    plt.draw()
    plt.pause(0.00000000001)


def save_plot():
    # save plot we've been building upon exit.
    # save to a file in pkg/plots directory.
    if fname != "":
        plt.savefig(pkg_path+"/plots/"+fname+"_demo.png", format='png')

def get_true_pose(msg):
    # save the true traj to plot it along w cur state.
    global true_pose
    true_pose = msg.data

def get_true_map(msg):
    rospy.loginfo("Ground truth map received by plotting node.")
    # plot the true map to compare to estimates.
    global true_map
    lm_x = [msg.data[i] for i in range(1,len(msg.data),3)]
    lm_y = [msg.data[i] for i in range(2,len(msg.data),3)]
    true_map = [lm_x, lm_y]
    

def main():
    global pkg_path
    rospy.init_node('plotting_node')

    # find the filepath to this package.
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('data_pkg')

    # when the node exits, make the plot.
    atexit.register(save_plot)

    # subscribe to the current state.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    rospy.Subscriber("/state/pf", Float32MultiArray, get_pf_state, queue_size=1)

    # subscribe to ground truth.
    rospy.Subscriber("/truth/veh_pose",Float32MultiArray, get_true_pose, queue_size=1)
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