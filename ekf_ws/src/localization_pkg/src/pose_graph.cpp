#include "localization_pkg/filter.h"

// Some of this implementation is based on an example from GTSAM itself:
// https://github.com/borglab/gtsam/blob/develop/examples/Pose2SLAMExample.cpp

// init the PoseGraph.
PoseGraph::PoseGraph() {
    // set filter type.
    this->type = FilterChoice::POSE_GRAPH_SLAM;
    // initialize state distribution.
    this->x_t.resize(3);
}

void PoseGraph::readParams(YAML::Node config) {
    // setup all commonly-used params.
    Filter::readCommonParams(config);
    // setup all filter-specific params, if any.
    std::string filter_to_compare_str = config["pose_graph"]["filter_to_compare"].as<std::string>();
    if (filter_to_compare_str == "ekf_slam") {
        this->filter_to_compare = FilterChoice::EKF_SLAM;
    } else if (filter_to_compare_str == "ukf_slam") {
        this->filter_to_compare = FilterChoice::UKF_SLAM;
    } else if (filter_to_compare_str == "ukf_loc") {
        this->filter_to_compare = FilterChoice::UKF_LOC;
    } else if (filter_to_compare_str == "naive") {
        this->filter_to_compare = FilterChoice::NAIVE_COMMAND_PROPAGATION;
    } else {
        throw std::runtime_error("Invalid choice of pose_graph.filter_to_compare in params.yaml.");
    }
    this->graph_size_threshold = config["pose_graph"]["graph_size_threshold"].as<int>();
    this->iteration_error_threshold = config["pose_graph"]["iteration_error_threshold"].as<float>();
    this->max_iterations = config["pose_graph"]["max_iterations"].as<int>();
    this->verbose = config["pose_graph"]["verbose"].as<bool>();

    // define noise models for both types of connections.
    this->process_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(this->V(0,0), this->V(0,0), this->V(1,1)));
    ///\note: BearingRangeFactor has bearing first and range second, so we swap the usual order for this project.
    this->sensing_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(this->W(1,1), this->W(0,0)));
}

void PoseGraph::init(float x_0, float y_0, float yaw_0) {
    // Ensure timesteps start at 0.
    this->timestep = 0;
    // Set the current veh pose.
    this->cur_veh_pose_estimate = gtsam::Pose2(x_0, y_0, yaw_0);

    // Add this pose as the first node in the factor graph.
    ROS_INFO_STREAM("PGS: Adding initial vehicle pose as first naive estimate to the pose graph.");
    this->initial_estimate.insert(timestep_to_veh_pose_key(this->timestep), this->cur_veh_pose_estimate);

    // We must assign some noise model for the prior.
    // auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
    ///\note: Since we are certain the initial pose is correct (since it is directly used as the origin for everything in the project), there is no noise assigned to it. This ensures that during optimization, the initial pose cannot be changed.
    auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.0, 0.0, 0.0));

    // Set this pose as the prior for our factor graph.
    ROS_INFO_STREAM("PGS: Adding prior for initial veh pose.");
    graph.addPrior(timestep_to_veh_pose_key(this->timestep), this->cur_veh_pose_estimate, priorNoise);

    // Set initialized flag.
    this->isInit = true;
}

void PoseGraph::onLandmarkMeasurement(int id, float range, float bearing) {
    ///////////////// Determine landmark ID/index /////////////////////
    int lm_index = -1;
    if (this->landmark_id_is_known) {
        // Check if the landmark with this ID has been detected before.
        for (int i=0; i < this->M; ++i) {
            if (this->lm_IDs[i] == id) {
                // We have detected this landmark before, so note the index.
                lm_index = i;
                break;
            }
        }
    } else {
        throw std::runtime_error("PGS with unknown landmark ID is not yet supported.");
    }

    /////////// Add landmark itself (if needed) ////////////////////
    if (lm_index == -1) {
        // This is the first detection of this landmark. Add its ID to the list.
        this->lm_IDs.push_back(id);
        lm_index = this->M;
        this->M++;

        // Use GTSAM's datatypes to convert this measurement into a transformation matrix.
        gtsam::Pose2 veh_to_landmark = gtsam::Pose2(range, 0.0, bearing);
        // Compute this landmark's global position.
        gtsam::Pose2 global_landmark_pose = veh_to_landmark * this->cur_veh_pose_estimate;
        gtsam::Vector3 global_landmark_position_3 = gtsam::Pose2::Logmap(global_landmark_pose);
        gtsam::Vector2 global_landmark_position_2 = gtsam::Vector2(global_landmark_position_3(0), global_landmark_position_3(1));
        ///\todo: This gives completely wrong landmark positions.
        // Use my less fancy way that definitely works.
        global_landmark_position_2(0) = this->x_t(0) + range*cos(this->x_t(2)+bearing);
        global_landmark_position_2(1) = this->x_t(1) + range*sin(this->x_t(2)+bearing);

        // Add this landmark as a node in the graph.
        if (this->verbose) {
            ROS_INFO_STREAM("PGS: Adding naive estimate for landmark " << id);
        }
        this->initial_estimate.insert(landmark_id_to_key(id), global_landmark_position_2);
    }

    ///////////////// Add vehicle->landmark connection /////////////////////////
    // Add a connection to the graph between the current vehicle pose and the detected landmark.
    ///\note: gtsam::Vector2 and gtsam::Point2 are identical.
    this->graph.emplace_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Vector2>>(timestep_to_veh_pose_key(this->timestep), landmark_id_to_key(id), gtsam::Rot2(bearing), range, this->sensing_noise_model);
    // this->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2> >(timestep_to_veh_pose_key(this->timestep), landmark_id_to_key(id), veh_to_landmark, this->sensing_noise_model);
    // Record this connection so we don't have to re-interpret the factor graph every iteration to generate state messages.
    this->msg_measurement_connections.push_back(this->timestep);
    this->msg_measurement_connections.push_back(lm_index);
}

// Update the graph with the new information for this iteration.
void PoseGraph::update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // check stopping criteria.
    if (this->solved_pose_graph) {
        // we already ran optimization, so just exit.
        // ROS_WARN_STREAM("PGS: Already solved pose graph, so just exiting immediately.");
        return;
    }

    ///\todo: maybe should have a better way to initiate all this besides a preset increment amount.
    if (this->timestep >= this->graph_size_threshold) {
        // run optimization algorithm and then exit.
        solvePoseGraph();
        // Publish the pose graph before and after optimization so the plotting_node can visualize the difference.
        publishState();
        return;
    }

    // use commanded motion to create a connection between previous and new vehicle pose.
    if (this->verbose) {
        ROS_INFO_STREAM("PGS: Adding command connection.");
    }
    this->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2> >(timestep_to_veh_pose_key(this->timestep), timestep_to_veh_pose_key(this->timestep+1), gtsam::Pose2(cmdMsg->fwd, 0, cmdMsg->ang), this->process_noise_model);

    // update timestep (i.e., iteration index).
    this->timestep += 1;

    // add the current naive pose estimate as a node in the pose graph.
    if (this->verbose) {
        ROS_INFO_STREAM("PGS: Adding naive estimate as a new node.");
    }
    this->initial_estimate.insert(timestep_to_veh_pose_key(this->timestep), this->cur_veh_pose_estimate);

    // process all measurements for this iteration, and generate loop closure constraints if applicable.
    std::vector<float> lm_meas = lmMeasMsg->data;
    int num_landmarks = (int) (lm_meas.size() / 3);
    // if there is at least one detection, handle each individually.
    for (int l=0; l<num_landmarks; ++l) {
        // extract the landmark details and create a graph connection.
        int id = (int)lm_meas[l*3]; // will only be used if this->landmark_ID_is_known. otherwise, function will do data association.
        float r = lm_meas[l*3+1];
        float b = lm_meas[l*3+2];
        onLandmarkMeasurement(id, r, b);
    }

    // Publish the progress building the pose graph so far.
    publishState();
}

void PoseGraph::solvePoseGraph() {
    ROS_INFO_STREAM("PGS: Attempting to solve the pose graph.");
    if (this->verbose) {
        // print the pose graph and our initial estimate of all vehicle poses.
        this->graph.print("\nFactor Graph:\n");
        this->initial_estimate.print("\nInitial Estimate:\n");
    }

    // Optimize the initial values using a Gauss-Newton nonlinear optimizer
    // The optimizer accepts an optional set of configuration parameters.
    // gtsam::GaussNewtonParams parameters;
    // parameters.relativeErrorTol = this->iteration_error_threshold;
    // parameters.maxIterations = this->max_iterations;
    // // Create the optimizer instance and run it.
    // gtsam::GaussNewtonOptimizer optimizer(this->graph, this->initial_estimate, parameters);
    // Create the optimizer instance with default params, and run it.
    gtsam::LevenbergMarquardtOptimizer optimizer(this->graph, this->initial_estimate);
    ROS_INFO_STREAM("PGS: Defined optimizer.");
    this->result = optimizer.optimize();
    ROS_INFO_STREAM("PGS: Ran optimization.");

    if (this->verbose) {
        this->result.print("Final Result:\n");
    
        // Calculate and print marginal covariances for all variables
        std::cout.precision(3);
        gtsam::Marginals marginals(graph, this->result);
        for (int i = 0; i < this->timestep; ++i) {
            std::cout << "covariance of vehicle pose " << i << ":\n" << marginals.marginalCovariance(i) << std::endl;
        }
    }
    ROS_INFO_STREAM("PGS: Finished solving pose graph.");

    this->solved_pose_graph = true;
}

void PoseGraph::publishState() {
    // Convert the entire pose graph to a ROS message and publish it.
    base_pkg::PoseGraphState stateMsg;
    // Set params that are the same for the pose graph both before and after optimization.
    stateMsg.timestep = this->timestep; // number of iterations before the pose graph was solved.
    stateMsg.M = this->M; // number of landmarks detected.
    ///\todo: encode measurement connections.
    stateMsg.meas_connections = this->msg_measurement_connections;

    // Set params specific to pose graph BEFORE optimization.
    // vehicle pose history before pose-graph optimization was run.
    std::vector<float> x_vec_initial;
    std::vector<float> y_vec_initial;
    std::vector<float> yaw_vec_initial;
    for (int i=0; i<this->timestep; ++i) {
        // get pose corresponding to iteration i.
        gtsam::Pose2 veh_pose_i = this->initial_estimate.at<gtsam::Pose2>(timestep_to_veh_pose_key(i));
        x_vec_initial.push_back(veh_pose_i.x());
        y_vec_initial.push_back(veh_pose_i.y());
        yaw_vec_initial.push_back(veh_pose_i.theta());
    }
    stateMsg.x_v = x_vec_initial;
    stateMsg.y_v = y_vec_initial;
    stateMsg.yaw_v = yaw_vec_initial;

    // all landmark positions before pose-graph optimization was run.
    std::vector<float> lm_initial;
    for (int i=0; i<this->M; ++i) {
        int j = this->lm_IDs[i]; // Convert from landmark index to ID.
        gtsam::Vector2 lm_pos_j = this->initial_estimate.at<gtsam::Vector2>(landmark_id_to_key(j));
        lm_initial.push_back(lm_pos_j(0));
        lm_initial.push_back(lm_pos_j(1));
    }
    stateMsg.landmarks = lm_initial;

    // publish this as the initial pose graph.
    this->statePubSecondary.publish(stateMsg);

    // if we've solved the pose graph, publish the result too.
    if (this->solved_pose_graph) {
        // replace the vehicle pose history with the solution to the pose graph optimization.
        std::vector<float> x_vec_result;
        std::vector<float> y_vec_result;
        std::vector<float> yaw_vec_result;
        for (int i=0; i<this->timestep; ++i) {
            // get pose corresponding to iteration i.
            gtsam::Pose2 veh_pose_i = this->result.at<gtsam::Pose2>(timestep_to_veh_pose_key(i));
            x_vec_result.push_back(veh_pose_i.x());
            y_vec_result.push_back(veh_pose_i.y());
            yaw_vec_result.push_back(veh_pose_i.theta());
        }
        stateMsg.x_v = x_vec_result;
        stateMsg.y_v = y_vec_result;
        stateMsg.yaw_v = yaw_vec_result;

        // all landmark positions after pose-graph optimization was run.
        std::vector<float> lm_result;
        for (int i=0; i<this->M; ++i) {
            int j = this->lm_IDs[i]; // Convert from landmark index to ID.
            gtsam::Vector2 lm_pos_j = this->initial_estimate.at<gtsam::Vector2>(landmark_id_to_key(j));
            lm_result.push_back(lm_pos_j(0));
            lm_result.push_back(lm_pos_j(1));
        }
        stateMsg.landmarks = lm_result;

        // publish it.
        this->statePub.publish(stateMsg);
    }
}