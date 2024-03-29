#!/usr/bin/env python3

"""
Create a live plot of the true and estimated states.
"""

import rospy, rospkg, yaml, sys
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from base_pkg.msg import EKFState, UKFState, PoseGraphState, NaiveState
import numpy as np
import atexit
from math import cos, sin, pi, atan2, remainder, tau, sqrt
from cv_bridge import CvBridge
bridge = CvBridge()

from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
# Suppress constant matplotlib warnings about thread safety.
import warnings
warnings.filterwarnings("ignore")

############ GLOBAL VARIABLES ###################
# store all plots objects we want to be able to remove later.
plots = {}
# publish clicked point on map for planner.
goal_pub = None
# points clicked on the map, if using LIST_CLICKED_POINTS mode.
clicked_points = []
# queue of true poses so we only show the one corresponding to the current estimate.
true_poses = []
# full history of filter estimates for veh pose so far. used to show estimate history compared to pose graph.
veh_pose_est_hist = { "x": [], "y": [], "dx": [], "dy": [], "time": []}
# average error from truth for full vehicle history of all filters running.
avg_errs = {}
###### Messages that have arrived and need to be added to the plot.
msg_ekf = None
msg_ukf = None
msg_pose_graph_init = None
msg_pose_graph_result = None
msg_naive = None
# figure objects that may or may not be defined.
sim_viz_fig = None
pose_graph_fig = None
#################################################

################# STATE CALLBACKS ####################
# get the state published by the filter, and queue it up for the plot to be updated.
def get_ekf_state(msg):
    global msg_ekf; msg_ekf = msg
def get_ukf_state(msg):
    global msg_ukf; msg_ukf = msg
def get_pose_graph_initial(msg):
    global msg_pose_graph_init; msg_pose_graph_init = msg
def get_pose_graph_result(msg):
    global msg_pose_graph_result; msg_pose_graph_result = msg
def get_naive_state(msg):
    global msg_naive; msg_naive = msg

################ OTHER CALLBACKS #######################
def get_true_pose(msg):
    # save the messages to a queue so they can be shown with the corresponding estimate.
    global true_poses; true_poses.append(msg)

def get_true_landmark_map(msg):
    rospy.loginfo("PLT: Ground truth landmark map received by plotting node.")
    lm_x = [msg.data[i] for i in range(1,len(msg.data),3)]
    lm_y = [msg.data[i] for i in range(2,len(msg.data),3)]
    # plot the true landmark positions to compare to estimates.
    if sim_viz_fig is not None:
        sim_viz_fig.scatter(lm_x, lm_y, s=30, color="white", edgecolors="black", zorder=2, label="True Landmark Position")
    if pose_graph_fig is not None:
        pose_graph_fig.scatter(lm_x, lm_y, s=30, color="white", edgecolors="black", zorder=1, label="True Landmark Position")

def get_color_map(msg):
    # get the true occupancy grid map image.
    color_map = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    # add the true map image to the plot. extent=(L,R,B,T) gives display bounds.
    edge = config["map"]["bound"]
    if sim_viz_fig is not None:
        sim_viz_fig.imshow(color_map, zorder=0, extent=[-edge, edge, -edge, edge])
    if pose_graph_fig is not None:
        pose_graph_fig.imshow(color_map, zorder=0, extent=[-edge, edge, -edge, edge])

def get_planned_path(msg):
    global plots
    # get the set of points that A* determined to get to the clicked goal.
    # remove and update the path if we've drawn it previously.
    remove_plot("planned_path")
    remove_plot("goal_pt")
    # only draw if the path is non-empty.
    if len(msg.data) > 1:
        plots["planned_path"] = sim_viz_fig.scatter([msg.data[i] for i in range(0,len(msg.data),2)], [msg.data[i] for i in range(1,len(msg.data),2)], s=12, color="purple", zorder=1, label="Planned Path")
        # show the goal point (end of path).
        plots["goal_pt"] = sim_viz_fig.scatter(msg.data[-2], msg.data[-1], color="yellow", edgecolors="black", s=40, zorder=2, label="Goal Point")

############ HELPER FUNCTIONS ######################
def remove_plot(name):
    """
    Remove the plot if it already exists. Otherwise do nothing.
    """
    if name in plots.keys():
        try:
            # this works for stuff like scatter(), arrow()
            plots[name].remove()
        except:
            # the first way doesn't work for plt.plot()
            line = plots[name].pop(0)
            line.remove()
        # remove the key.
        del plots[name]

def save_plot(base_pkg_path):
    # save plot we've been building upon exit.
    # save to a file in base_pkg/plots directory.
    if config["plotter"]["save_final_map"]:
        fpath = base_pkg_path+"/plots/"+config["filter"]+"_demo.png"
        rospy.logwarn("PLT: Saving plot to " + fpath)
        plt.savefig(fpath, format='png')

    # save avg err of filters to respective files in base_pkg/data directory.
    if config["pose_graph"]["save_average_error_at_end"] and len(avg_errs.keys()) > 0:
        for type in avg_errs.keys():
            with open(base_pkg_path+'/data/{:}.csv'.format(type), 'a') as file:
                file.write("{:}\n".format(avg_errs[type]))

def on_click(event):
    # global clicked_points
    if event.button is MouseButton.RIGHT:
        # kill the node.
        rospy.logwarn("PLT: Killing plotting_node on right click.")
        exit()
    elif event.button is MouseButton.LEFT:
        if config["plotter"]["list_clicked_points"]:
            clicked_points.append((event.xdata, event.ydata))
            print(clicked_points)
        # publish new goal pt for the planner.
        goal_pub.publish(Vector3(x=event.xdata, y=event.ydata))
        # show this point so it appears even when not using A* planning.
        get_planned_path(Float32MultiArray(data=[event.xdata, event.ydata]))

def cov_to_ellipse(P_v):
    """
    Given the state covariance matrix, compute points to plot representative ellipse.
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

# https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def get_legend_symbol(shape:str, color:str):
    """
    Get a legend handle for the requested shape and color.
    """
    if shape == "ellipse":
        return mpatches.Circle((0.5, 0.5), 0.25, facecolor="white", edgecolor=color, linewidth=3)
    elif shape == "arrow":
        return Line2D([], [], color=color, marker=">", markersize=12, linestyle="none")
    else:
        rospy.logerr("PLT: Requested legend symbol with invalid shape, {:}.".format(shape))
        exit()

def compute_average_error(type:str, x_est, y_est, timestamps=None):
    """
    Compute average error of the filter vs ground truth for the full vehicle history.
    @param type - identifier of what filter this data is from.
    @param x_est - list of estimated x position in meters for each timestep.
    @param y_est - list of estimated y position in meters for each timestep.
    @param timestamps (optional) - list of timestep numbers corresponding to the entries in x_est and y_est.
    """
    num_iters = len(x_est)
    # Extract true poses for comparison.
    if timestamps is None:
        true_x = [pose.x for pose in true_poses[:num_iters]]
        true_y = [pose.y for pose in true_poses[:num_iters]]
    else:
        true_x = [true_poses[i-1].x for i in timestamps]
        true_y = [true_poses[i-1].y for i in timestamps]
    # Compute position error for all timesteps.
    errors = [sqrt((x_est[i]-true_x[i])**2 + (y_est[i]-true_y[i])**2) for i in range(num_iters)]
    # Get average error.
    avg_err = sum(errors) / num_iters
    rospy.loginfo("PLT: Average error in {:} from true vehicle pose history = {:}.".format(type, avg_err))

    global avg_errs
    avg_errs[type] = avg_err


############### MAIN UPDATE LOOP ######################
def update_plot(event):
    """
    Update viz with any new messages that have arrived since the last iteration.
    """
    global plots
    global msg_ekf, msg_ukf, msg_pose_graph_init, msg_pose_graph_result, msg_naive

    # If we're only supposed to plot the final result, don't do anything until we actually have the final pose graph.
    if plot_result_only and config["filter"] == "pose_graph" and msg_pose_graph_result is None:
        return

    # NOTE we make a local copy of each and clear its global counterpart to avoid:
    # (a) the msg being changed while we're in the middle of plotting it, and 
    # (b) so we don't "update" the plot if a new message has not arrived since we last plotted it.
    
    # Only run one update function for each figure.
    msg = None
    type = None
    if sim_viz_fig is not None:
        if msg_naive is not None:
            msg = msg_naive
            msg_naive = None
            type = "naive"
        if msg_ekf is not None:
            msg = msg_ekf
            msg_ekf = None
            type = "ekf"
        if msg_ukf is not None:
            msg = msg_ukf
            msg_ukf = None
            type = "ukf"

    # If we're only supposed to plot the final result, don't do anything until we actually have a final result.
    if plot_result_only and msg is not None and msg.timestep + 1 < config["num_iterations"]:
        return
    
    if msg is not None:
        """
        Perform necessary plot updates for a new state message for the main sim viz figure.
        @param msg - EKFState, UKFState, or NaiveState message.
        @param type - either "ekf", "ukf", or "naive", specifying the message/filter type.
        """
        if type not in ["ekf", "ukf", "naive"]:
            rospy.logerr("PLT: update_kf_plot called with invalid type {:}.".format(type))
            return

        sim_viz_fig.title.set_text(type.upper()+" Estimate")
        #################### TIMESTEP ######################
        remove_plot("timestep")
        plots["timestep"] = sim_viz_fig.text(-config["map"]["bound"], config["map"]["bound"], 't = '+str(msg.timestep), horizontalalignment='left', verticalalignment='bottom', zorder=2)

        #################### TRUE POSE #########################
        if config["plotter"]["show_true_traj"] and msg.timestep <= len(true_poses):
            pose = true_poses[msg.timestep-1]
            # plot the current veh pos, & remove previous.
            remove_plot("veh_pos_true")
            plots["veh_pos_true"] = sim_viz_fig.arrow(pose.x, pose.y, config["plotter"]["arrow_len"]*cos(pose.z), config["plotter"]["arrow_len"]*sin(pose.z), color="blue", width=0.1, zorder=2)

        ################ VEH POS #################
        # plot current estimated veh pos.
        if not config["plotter"]["show_entire_traj"]:
            remove_plot("veh_pos_est")
        # draw a single pt with arrow to represent current veh pose.
        dx = config["plotter"]["arrow_len"]*cos(msg.yaw_v)
        dy = config["plotter"]["arrow_len"]*sin(msg.yaw_v)
        plots["veh_pos_est"] = sim_viz_fig.arrow(msg.x_v, msg.y_v, dx, dy, facecolor="green", width=0.1, zorder=4, edgecolor="black")
        # keep track of full vehicle pose history.
        global veh_pose_est_hist
        veh_pose_est_hist["time"].append(msg.timestep)
        veh_pose_est_hist["x"].append(msg.x_v)
        veh_pose_est_hist["y"].append(msg.y_v)
        veh_pose_est_hist["dx"].append(dx)
        veh_pose_est_hist["dy"].append(dy)
        # only plot it if the pose graph is being continuously solved so it doesn't have the initial estimate.
        if config["pose_graph"]["solve_graph_every_iteration"] and pose_graph_fig is not None:
            # draw the secondary filter's pose estimate history under the pose graphs for comparison.
            remove_plot("pg_veh_pos_est")
            plots["pg_veh_pos_est"] = pose_graph_fig.quiver(veh_pose_est_hist["x"], veh_pose_est_hist["y"], veh_pose_est_hist["dx"], veh_pose_est_hist["dy"], color="green", width=0.1, zorder=4, pivot="mid", minlength=0.0001)
        # compute error from ground truth.
        if msg.timestep >= config["num_iterations"]:
            compute_average_error(type, veh_pose_est_hist["x"], veh_pose_est_hist["y"], veh_pose_est_hist["time"])

        if type in ["ekf", "ukf"]:
            # compute length of state to use throughout. n = 3+2M
            n = int(len(msg.P)**(1/2))
            ################ VEH COV ##################
            if config["plotter"]["show_veh_ellipse"]:
                # compute parametric ellipse for veh covariance.
                veh_ell = cov_to_ellipse(np.array([[msg.P[0],msg.P[1]], [msg.P[n],msg.P[n+1]]]))
                # remove old ellipses.
                if not config["plotter"]["show_entire_traj"]:
                    remove_plot("veh_cov_est")
                # plot the ellipse.
                plots["veh_cov_est"], = sim_viz_fig.plot(msg.x_v+veh_ell[0,:] , msg.y_v+veh_ell[1,:],'lightgrey', zorder=1)

            ############## LANDMARK EST ##################
            lm_x = [msg.landmarks[i] for i in range(1,len(msg.landmarks),3)]
            lm_y = [msg.landmarks[i] for i in range(2,len(msg.landmarks),3)]
            # remove old landmark estimates.
            remove_plot("lm_pos_est")
            # plot new landmark estimates.
            plots["lm_pos_est"] = sim_viz_fig.scatter(lm_x, lm_y, s=30, color="red", edgecolors="black", zorder=3, label="Estimated Landmark Position")

            ############## LANDMARK COV ###################
            if config["plotter"]["show_landmark_ellipses"]:
                # plot new landmark covariances.
                for i in range(len(msg.landmarks) // 3):
                    # replace previous if it's been plotted before.
                    lm_id = msg.landmarks[i*3]
                    remove_plot("lm_cov_est_{:}".format(i))
                    # extract 2x2 cov for this landmark.
                    lm_ell = cov_to_ellipse(np.array([[msg.P[3+2*i],msg.P[4+2*i]],[msg.P[n+3+2*i],msg.P[n+4+2*i]]]))
                    # plot its ellipse.
                    plots["lm_cov_est_{:}".format(i)], = sim_viz_fig.plot(lm_x[i]+lm_ell[0,:], lm_y[i]+lm_ell[1,:], 'orange', zorder=1)
        
        ############## UKF SIGMA POINTS ##################
        if type == "ukf":
            # extract length of vectors in sigma pts (not necessarily = n).
            n_sig = int(((1+8*len(msg.X))**(1/2) - 1) / 4) # soln to n*(2n+1)=len.
            # check if we're using UKF with 3x1 or 4x1 veh state.
            veh_len = 3 if n_sig % 2 == 1 else 4

            ############## VEH POSE SIGMA POINTS #################
            # only show sigma points' veh pose, not landmark info.
            if config["plotter"]["plot_ukf_arrows"]:
                # TODO change this to use quiver, and add symbol to legend handles.
                # plot as arrows (slow).
                # draw a pt with arrow for all sigma pts.
                for i in range(0, 2*n_sig+1):
                    remove_plot("veh_sigma_pts_{:}".format(i))
                    yaw = msg.X[i*n_sig+2] if veh_len == 3 else remainder(atan2(msg.X[i*n_sig+3], msg.X[i*n_sig+2]), tau)
                    plots["veh_sigma_pts_{:}".format(i)] = sim_viz_fig.arrow(msg.X[i*n_sig], msg.X[i*n_sig+1], config["plotter"]["arrow_len"]*cos(yaw), config["plotter"]["arrow_len"]*sin(yaw), color="cyan", width=0.1)
            else: # just show x,y of pts.
                X_x = [msg.X[i*n_sig] for i in range(0,2*n_sig+1)]
                X_y = [msg.X[i*n_sig+1] for i in range(0,2*n_sig+1)]
                # remove old points.
                remove_plot("veh_sigma_pts")
                # plot sigma points.
                plots["veh_sigma_pts"] = sim_viz_fig.scatter(X_x, X_y, s=30, color="tab:cyan", zorder=2)

            ################# LANDMARK SIGMA POINTS ##################
            if config["plotter"]["show_landmark_sigma_pts"]:
                # plot all landmark sigma pts.
                X_lm_x = []; X_lm_y = []
                for j in range(2*n_sig+1):
                    X_lm_x += [msg.X[j*n_sig+i] for i in range(veh_len,n_sig,2)]
                    X_lm_y += [msg.X[j*n_sig+i+1] for i in range(veh_len,n_sig,2)]
                # remove old points.
                remove_plot("lm_sigma_pts")
                # plot sigma points.
                plots["lm_sigma_pts"] = sim_viz_fig.scatter(X_lm_x, X_lm_y, s=30, color="tab:cyan", zorder=1, label="UKF Landmark Sigma Points")
                
    msg = None
    type = None
    if pose_graph_fig is not None:
        if msg_pose_graph_init is not None:
            msg = msg_pose_graph_init
            msg_pose_graph_init = None
            type = "init"
            # # Compute error from ground truth for initial vehicle estimate.
            # if not config["pose_graph"]["solve_graph_every_iteration"] and msg.timestep + 1 >= config["num_iterations"]:
            #     compute_average_error("pose_graph_{:}".format(type), msg.x_v, msg.y_v)
        if msg_pose_graph_result is not None:
            msg = msg_pose_graph_result
            msg_pose_graph_result = None
            type = "result"

    # If we're only supposed to plot the final result, don't do anything until we actually have a final result.
    if plot_result_only and msg is not None and msg.timestep + 1 < config["num_iterations"]:
        return

    if msg is not None:
        """
        Plot the graph encoded in a PoseGraphState message for a new factor graph.
        @param msg - PoseGraphState message.
        @param type - either "init" or "result", specifying the message/filter type.
        """
        if type not in ["init", "result"]:
            rospy.logerr("PLT: update_pose_graph_plot called with invalid type {:}.".format(type))
            return
        
        # Some plot params will be different depending on the data type.
        pgs_title = {"init" : "Pose-Graph Before Optimization", "result" : "Pose-Graph After Optimization"}
        pgs_veh_color = {"init" : "green", "result" : "purple"}
        pgs_lm_color = {"init" : "red", "result" : "darkred"}
        pgs_lm_label = {"init" : "Estimated Landmark Position", "result" : "Pose-Graph SLAM Result (Landmarks)"}
        # TODO may want to plot result graph on top of init graph, so parametrize zorders by type.

        # set title.
        pose_graph_fig.title.set_text(pgs_title[type])

        #################### TRUE POSE #########################
        if config["plotter"]["show_true_traj"] and msg.timestep <= len(true_poses):
            # Clear last set of poses. (Could simply append a new pose each time but then the number of plots scales with number of iterations rather than remaining constant.)
            remove_plot("pg_veh_pos_true")
            # Compute components of set of true poses up to now.
            true_x = [pose.x for pose in true_poses[:msg.timestep]]
            true_y = [pose.y for pose in true_poses[:msg.timestep]]
            true_yaw = [pose.z for pose in true_poses[:msg.timestep]]
            arrow_x_components = [config["plotter"]["arrow_len"]*cos(true_yaw[i]) for i in range(msg.timestep)]
            arrow_y_components = [config["plotter"]["arrow_len"]*sin(true_yaw[i]) for i in range(msg.timestep)]
            # plot all true veh poses up to now.
            plots["pg_veh_pos_true"] = pose_graph_fig.quiver(true_x, true_y, arrow_x_components, arrow_y_components, color="blue", width=0.1, zorder=1, pivot="mid", minlength=0.0001)
        
        ############# VEHICLE POSE HISTORY ###################
        remove_plot(type+"_pg_veh_pose_history")
        arrow_x_components = [config["plotter"]["arrow_len"]*cos(msg.yaw_v[i]) for i in range(msg.timestep)]
        arrow_y_components = [config["plotter"]["arrow_len"]*sin(msg.yaw_v[i]) for i in range(msg.timestep)]
        plots[type+"_pg_veh_pose_history"] = pose_graph_fig.quiver(msg.x_v, msg.y_v, arrow_x_components, arrow_y_components, color=pgs_veh_color[type], width=0.1, zorder=5, edgecolor="black", pivot="mid", minlength=0.0001) # add argument linewidth=1 to show border around each pose.

        # Compute error from ground truth.
        if msg.timestep + 1 >= config["num_iterations"]:
            compute_average_error("pose_graph_{:}".format(type), msg.x_v, msg.y_v)

        ############### LANDMARK POSITIONS ###################
        remove_plot(type+"_pg_landmarks")
        if msg.M > 0: # verify there is at least one landmark.
            lm_x = [msg.landmarks[i] for i in range(0,2*msg.M,2)]
            lm_y = [msg.landmarks[i] for i in range(1,2*msg.M,2)]
            plots[type+"_pg_landmarks"] = pose_graph_fig.scatter(lm_x, lm_y, s=30, color=pgs_lm_color[type], edgecolors="black", zorder=2, label=pgs_lm_label[type])

        ############### ADJACENT POSE CONNECTIONS ###############
        if config["plotter"]["pose_graph"]["show_cmd_connections"]:
            # we know a connection exists between every vehicle pose and the pose on the immediate previous/next iterations.
            remove_plot(type+"_pg_cmd_connections")
            plots[type+"_pg_cmd_connections"] = pose_graph_fig.plot(msg.x_v, msg.y_v, color="blue", zorder=0, label="Pose-Graph Command Connection")

        ############## MEASUREMENT CONNECTIONS ####################
        if config["plotter"]["pose_graph"]["show_meas_connections"]:
            for j in range(len(msg.meas_connections) // 2):
                i_veh = msg.meas_connections[2*j] - 1 # vehicle index is one lower than its iteration number.
                i_lm = msg.meas_connections[2*j+1] # landmark index.
                # plot a line between the specified vehicle pose and landmark.
                remove_plot(type+"_pg_meas_connection_{:}".format(j))
                plots[type+"_pg_meas_connection_{:}".format(j)] = pose_graph_fig.plot([msg.x_v[i_veh], lm_x[i_lm]], [msg.y_v[i_veh], lm_y[i_lm]], color="lightcoral", zorder=0, label="Pose-Graph Measurement Connection")
    
    # force desired window region (prevents axes expanding when vehicle comes close to an edge of the plot).
    plt.xlim(display_region)
    plt.ylim(display_region)

    # Make a legend for the overall plot.
    remove_plot("leg")
    # Collect legend handles for all subplots.
    handles, labels = plt.gca().get_legend_handles_labels()
    # Manually add legend entries for stuff like arrows and ellipses that don't show up right.
    if "veh_pos_true" in plots.keys():
        handles.extend([get_legend_symbol("arrow", "blue")])
        labels.extend(["True Vehicle Pose"])
    if "veh_pos_est" in plots.keys():
        handles.extend([get_legend_symbol("arrow", "green")])
        labels.extend(["Estimated Vehicle Pose"])
    if "veh_cov_est" in plots.keys():
        handles.extend([get_legend_symbol("ellipse", "lightgrey")])
        labels.extend(["Vehicle Estimate Covariance"])
    if "lm_cov_est_0" in plots.keys():
        # There is at least one landmark whose covariance has been plotted.
        handles.extend([get_legend_symbol("ellipse", "orange")])
        labels.extend(["Landmark Estimate Covariance"])
    if "result_pg_veh_pose_history" in plots.keys():
        handles.extend([get_legend_symbol("arrow", "purple")])
        labels.extend(["Pose-Graph SLAM Result (Vehicle Poses)"])
    # Run through a dictionary to remove duplicates.
    by_label = dict(zip(labels, handles))
    # Create the legend itself.
    plots["leg"] = fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=3, handler_map={mpatches.Circle: HandlerEllipse()})

    # Update the viz with our changes.
    plt.draw()
    plt.pause(0.00000000001)


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

    # read command line args.
    global dt, plot_result_only
    if len(sys.argv) < 3:
        rospy.logerr("Required params (timer_period, plot_result_only) not provided to plotting_node.")
        exit()
    else:
        if sys.argv[1].lower() == "default":
            # Read dt from config file.
            dt = config["dt"]
        else:
            # Use the provided value.
            dt = float(sys.argv[1])
        plot_result_only = sys.argv[2].lower() == "true"

    # make live plot bigger.
    plt.rcParams["figure.figsize"] = (9,9)

    # when the node exits, make the plot.
    atexit.register(save_plot, base_pkg_path)

    # subscribe to the current state for all filters.
    rospy.Subscriber("/state/ekf", EKFState, get_ekf_state, queue_size=1)
    rospy.Subscriber("/state/ukf", UKFState, get_ukf_state, queue_size=1)
    if not config["pose_graph"]["solve_graph_every_iteration"]:
        # Don't bother updating two pose graphs since they'll be identical, and we'll be getting the "result" graph in our callback every iteration.
        rospy.Subscriber("/state/pose_graph/initial", PoseGraphState, get_pose_graph_initial, queue_size=1)
    rospy.Subscriber("/state/pose_graph/result", PoseGraphState, get_pose_graph_result, queue_size=1)
    rospy.Subscriber("/state/naive", NaiveState, get_naive_state, queue_size=1)

    # Only setup som subscribers if config params request it.
    # subscribe to ground truth.
    if config["plotter"]["show_true_traj"]:
        rospy.Subscriber("/truth/veh_pose", Vector3, get_true_pose, queue_size=1)
    if config["plotter"]["show_true_landmark_map"]:
        rospy.Subscriber("/truth/landmarks", Float32MultiArray, get_true_landmark_map, queue_size=1)

    # subscribe to the color map.
    if config["plotter"]["show_occ_map"]:
        rospy.Subscriber("/truth/color_map", Image, get_color_map, queue_size=1)

    # publish the chosen goal point for the planner.
    goal_pub = rospy.Publisher("/plan/goal", Vector3, queue_size=1)
    # subscribe to planned path to the goal.
    rospy.Subscriber("/plan/path", Float32MultiArray, get_planned_path, queue_size=1)

    # startup the plot.
    global fig
    fig = plt.figure()
    global sim_viz_fig, pose_graph_fig
    if config["filter"].lower() == "pose_graph":
        if config["plotter"]["pose_graph"]["show_normal_viz_alongside"]:
            sim_viz_fig = plt.subplot(1, 2, 1)
            plt.title("Ground truth and online filter progress")
            # force desired window region.
            plt.xlim(display_region)
            plt.ylim(display_region)
            pose_graph_fig = plt.subplot(1, 2, 2)
            plt.title("Pose graph progress")
            # force desired window region.
            plt.xlim(display_region)
            plt.ylim(display_region)
            sim_viz_fig.set_aspect('equal')
            pose_graph_fig.set_aspect('equal')
        else: # only show pose graph.
            pose_graph_fig = plt.subplot(1, 1, 1)
            plt.axis("equal")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            
    else:
        # Will not be plotting a pose graph, so only open one plot window.
        sim_viz_fig = plt.subplot(1, 1, 1)
        plt.axis("equal")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
    
    # allow user to click on the plot to set the goal position or kill the node.
    plt.connect('button_press_event', on_click)

    # Setup a timer for the plot update loop.
    rospy.Timer(rospy.Duration(dt), update_plot)

    # Start the plot display.
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
