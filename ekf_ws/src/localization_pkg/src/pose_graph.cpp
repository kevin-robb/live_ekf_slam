#include "localization_pkg/filter.h"

// Some of this implementation is based on an example from GTSAM itself:
// https://github.com/borglab/gtsam/blob/develop/examples/Pose2SLAMExample.cpp

// init the PoseGraph.
PoseGraph::PoseGraph() {
    // set filter type.
    this->type = FilterChoice::POSE_GRAPH_SLAM;
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
    std::string implementation_str = config["pose_graph"]["implementation"].as<std::string>();
    if (implementation_str == "gtsam") {
        this->impl_to_use = PoseGraphSlamImplementation::GTSAM;
    } else if (implementation_str == "sesync") {
        this->impl_to_use = PoseGraphSlamImplementation::SESYNC;
        throw std::runtime_error("SE-Sync pose graph SLAM implementation is incomplete. Choose another option in params.yaml.");
    } else if (implementation_str == "custom") {
        this->impl_to_use = PoseGraphSlamImplementation::CUSTOM;
        throw std::runtime_error("Custom pose graph SLAM implementation is incomplete. Choose another option in params.yaml.");
    } else {
        throw std::runtime_error("Invalid choice of pose_graph.implementation in params.yaml.");
    }

    this->num_iterations_total = config["num_iterations"].as<int>();
    this->verbose = config["pose_graph"]["verbose"].as<bool>();
    this->update_landmarks_after_adding = config["pose_graph"]["update_landmarks_after_adding"].as<bool>();
    this->solve_graph_every_iteration = config["pose_graph"]["solve_graph_every_iteration"].as<bool>();
    if (this->solve_graph_every_iteration) {
        // Don't let two different sources modify nodes in the initial_estimate values after adding them.
        this->update_landmarks_after_adding = false;
    }

    if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
        // Define noise models for both types of connections that will be created as we run.
        this->process_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(this->V(0,0), this->V(0,0), this->V(1,1)));
        ///\note: BearingRangeFactor has bearing first and range second, so we swap the usual order for this project.
        this->sensing_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector2(this->W(1,1), this->W(0,0)));
    } else if (this->impl_to_use == PoseGraphSlamImplementation::SESYNC) {
        this->sesync_options.verbose = this->verbose;
        // Options snagged from SE-Sync's main.cpp example.
        // Initialization method. Options are:  Chordal, Random.
        this->sesync_options.initialization = SESync::Initialization::Chordal;
        // Specific form of the synchronization problem to solve. Options are: Simplified, Explicit, SOSync.
        ///\note: Since we're hacking it to let landmarks be represented as SE(2) poses with a rotation weight of 0, we must use the translation-excplicit formulation of SE-Sync to avoid some 
        this->sesync_options.formulation = SESync::Formulation::Explicit;
        this->sesync_options.num_threads = 4;
        
    }
}

void PoseGraph::init(float x_0, float y_0, float yaw_0) {
    // Ensure timesteps start at 0.
    this->timestep = 0;
    // Set the current veh pose.
    this->x_t.resize(3);
    this->x_t << x_0, y_0, yaw_0;

if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
        this->cur_veh_pose_estimate = gtsam::Pose2(x_0, y_0, yaw_0);
        // Add this pose as the first node in the factor graph.
        ROS_INFO_STREAM("PGS: Adding initial vehicle pose as first naive estimate to the pose graph.");
        this->initial_estimate.insert(timestep_to_veh_pose_key(this->timestep), this->cur_veh_pose_estimate);
            
        // We must assign some noise model for the prior.
        // auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
        auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(1.3, 1.3, 1.2));
        ///\note: Since we are certain the initial pose is correct (since it is directly used as the origin for everything in the project), there is no noise assigned to it. This ensures that during optimization, the initial pose cannot be changed.
        // auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.0, 0.0, 0.0));

        // Set this pose as the prior for our factor graph.
        ROS_INFO_STREAM("PGS: Adding prior for initial veh pose.");
        graph.addPrior(timestep_to_veh_pose_key(this->timestep), this->cur_veh_pose_estimate, priorNoise);
    }


    // Set initialized flag.
    this->isInit = true;
}

void PoseGraph::updateNaiveVehPoseEstimate(Eigen::VectorXd state_vector, std::vector<int> landmark_ids) {
    // This estimate may come from another filter such as the EKF, or could be a basic propagation with no filtering.
    ///\note: This state vector is assumed to follow the EKF format of (x, y, yaw, landmark_x_1, landmark_y_1, ..., landmark_x_M, landmark_y_M).
    // (There might be no landmarks.)

    // Update our current naive belief of the vehicle pose.
    this->x_t = state_vector;
    // this->x_t << state_vector(0), state_vector(1), state_vector(2);

    if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
        // Save this estimate directly as a pose matrix.
        this->cur_veh_pose_estimate = gtsam::Pose2(state_vector(0), state_vector(1), state_vector(2));
        ///\note: This vehicle pose estimate will be added as a node in the pose graph during the update() loop.
            
        if (this->update_landmarks_after_adding) {
            // Update any nodes we've already added to the pose graph for landmark position estimates.
            for (int i = 0; i < this->M; ++i) {
                int id = landmark_ids[i];
                this->initial_estimate.update(landmark_id_to_key(id), gtsam::Vector2(state_vector(3 + 2*i), state_vector(4 + 2*i)));
            }
        }
    }
}


int PoseGraph::getLandmarkIndexFromID(int id) {
    // Determine landmark index given its ID.
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
    // Add landmark itself (if needed).
    if (lm_index == -1) {
        // This is the first detection of this landmark. Add its ID to the list.
        this->lm_IDs.push_back(id);
        // lm_index = this->M;
        this->M++;
    }
    // Return the landmark's index.
    return lm_index;
}


void PoseGraph::onLandmarkMeasurement(int id, float range, float bearing) {
    int lm_index = PoseGraph::getLandmarkIndexFromID(id);

    if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
        // If this is the first detection of this landmark, add it as a graph node.
        if (lm_index == -1) {
            // // Use GTSAM's datatypes to convert this measurement into a transformation matrix.
            // gtsam::Pose2 veh_to_landmark = gtsam::Pose2(range, 0.0, bearing);
            // // Compute this landmark's global position.
            // gtsam::Pose2 global_landmark_pose = veh_to_landmark * this->cur_veh_pose_estimate;
            // gtsam::Vector3 global_landmark_position_3 = gtsam::Pose2::Logmap(global_landmark_pose);
            // gtsam::Vector2 global_landmark_position = gtsam::Vector2(global_landmark_position_3(0), global_landmark_position_3(1));
            ///\todo: This gives completely wrong landmark positions.
            // Use my less fancy way that definitely works.
            gtsam::Vector2 global_landmark_position = gtsam::Vector2(this->x_t(0) + range*cos(this->x_t(2)+bearing), this->x_t(1) + range*sin(this->x_t(2)+bearing));

            // Add this landmark as a node in the graph.
            if (this->verbose) {
                ROS_INFO_STREAM("PGS: Adding naive estimate for landmark " << id);
            }
            this->initial_estimate.insert(landmark_id_to_key(id), global_landmark_position);
        }
    
        // Add a connection to the graph between the current vehicle pose and the detected landmark.
        ///\note: gtsam::Vector2 and gtsam::Point2 are identical.
        this->graph.emplace_shared<gtsam::BearingRangeFactor<gtsam::Pose2, gtsam::Vector2>>(timestep_to_veh_pose_key(this->timestep), landmark_id_to_key(id), gtsam::Rot2(bearing), range, this->sensing_noise_model);
        // Record this connection so we don't have to re-interpret the factor graph every iteration to generate state messages.
        this->msg_measurement_connections.push_back(this->timestep);
        this->msg_measurement_connections.push_back(lm_index);
    } else if (this->impl_to_use == PoseGraphSlamImplementation::SESYNC) {
        // SE-Sync doesn't care about the "nodes", only the connections.
        // Create a measurement to add to our list of relative poses.
        SESync::RelativePoseMeasurement measurement;
        // Pose ids.
        measurement.i = timestep_to_veh_pose_key(this->timestep);
        measurement.j = landmark_id_to_key(id);
        // Raw "measurements" (commanded motion).
        measurement.t = Eigen::Matrix<SESync::Scalar, 2, 1>(range, 0.0);
        measurement.R = Eigen::Rotation2D<SESync::Scalar>(bearing).toRotationMatrix();
        // Set desired precisions of each component, using our known noise model.
        measurement.tau = this->V(0,0); // translational.
        ///\note: Since landmarks are only positions (w/ no discernable orientation), we set the rotational component's weight to zero. It will still be estimated, but it won't affect anything else and we can simply ignore landmark orientations in the final result.
        measurement.kappa = 0.0; // rotational.
        // measurement.kappa = 10000000.0; // rotational.
        // Add the measurement to our list.
        this->sesync_measurements.push_back(measurement);
    }
}

// Update the graph with the new information for this iteration.
void PoseGraph::update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // check stopping criteria.
    if (this->solved_pose_graph && !this->solve_graph_every_iteration) {
        // we already ran optimization, so just exit.
        // ROS_WARN_STREAM("PGS: Already solved pose graph, so just exiting immediately.");
        return;
    }

    // offset timestep (starts at 0) to compare to total iterations.
    if (this->timestep+1 >= this->num_iterations_total) {
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
    if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
        // Create a connection in the pose graph.
        this->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2> >(timestep_to_veh_pose_key(this->timestep), timestep_to_veh_pose_key(this->timestep+1), gtsam::Pose2(cmdMsg->fwd, 0, cmdMsg->ang), this->process_noise_model);
    } else if (this->impl_to_use == PoseGraphSlamImplementation::SESYNC) {
        // Create a measurement to add to our list of relative poses.
        SESync::RelativePoseMeasurement measurement;
        // Pose ids.
        measurement.i = timestep_to_veh_pose_key(this->timestep);
        measurement.j = timestep_to_veh_pose_key(this->timestep + 1);
        // Raw "measurements" (commanded motion).
        measurement.t = Eigen::Matrix<SESync::Scalar, 2, 1>(cmdMsg->fwd, 0.0);
        measurement.R = Eigen::Rotation2D<SESync::Scalar>(cmdMsg->ang).toRotationMatrix();
        // Set desired precisions of each component, using our known noise model.
        measurement.tau = this->V(0,0); // translational.
        measurement.kappa = this->V(1,1); // rotational.
        // Add the measurement to our list.
        this->sesync_measurements.push_back(measurement);
    }

    // update timestep (i.e., iteration index).
    this->timestep += 1;

    // add the current naive pose estimate as a node in the pose graph.
    if (this->verbose) {
        ROS_INFO_STREAM("PGS: Adding naive estimate as a new node.");
    }
    if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
        this->initial_estimate.insert(timestep_to_veh_pose_key(this->timestep), this->cur_veh_pose_estimate);
    }

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

    if (this->solve_graph_every_iteration) {
        // Solve the pose graph.
        solvePoseGraph();
        // Use this iterative solution to update the "initial" belief of the graph.
        this->initial_estimate = this->result;
    }

    // Publish the progress building the pose graph so far.
    publishState();
}

void PoseGraph::solvePoseGraph() {
    if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
        if (this->verbose) {
            ROS_INFO_STREAM("PGS: Attempting to solve the pose graph using GTSAM.");
            // print the pose graph and our initial estimate of all vehicle poses.
            this->graph.print("\nFactor Graph:\n");
            this->initial_estimate.print("\nInitial Estimate:\n");
        }

        // Create the optimizer instance with default params, and run it.
        gtsam::LevenbergMarquardtOptimizer optimizer(this->graph, this->initial_estimate);
        this->result = optimizer.optimize();

        if (this->verbose) {
            this->result.print("Final Result:\n");
        
            // Calculate and print marginal covariances for all variables
            std::cout.precision(3);
            gtsam::Marginals marginals(graph, this->result);
            for (int i = 0; i < this->timestep; ++i) {
                std::cout << "covariance of vehicle pose " << i << ":\n" << marginals.marginalCovariance(i) << std::endl;
            }
        }
    } else if (this->impl_to_use == PoseGraphSlamImplementation::SESYNC) {
        ROS_INFO_STREAM("PGS: Attempting to solve the pose graph using SE-Sync.");
        this->sesync_result = SESync::SESync(this->sesync_measurements, this->sesync_options);
    }
    
    if (this->verbose) {
        ROS_INFO_STREAM("PGS: Finished solving pose graph.");
    }
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

    // Select a list of values to publish.
    gtsam::Values nodes_to_send = this->initial_estimate;
    if (this->solved_pose_graph) {
        nodes_to_send = this->result;
    }
    if (this->impl_to_use == PoseGraphSlamImplementation::SESYNC && !this->solved_pose_graph) {
        ///\todo: Make a different thing that checks SE-Sync measurements rather than results.
        return;
    }

    // vehicle pose history estimate.
    std::vector<float> x_vec;
    std::vector<float> y_vec;
    std::vector<float> yaw_vec;
    gtsam::Pose2 veh_pose_i;
    gtsam::Rot2 veh_rot;
    gtsam::Point2 veh_pos;
    for (int i=0; i<this->timestep; ++i) {
        // get pose corresponding to iteration i.
        if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
            // directly access by key.
            veh_pose_i = nodes_to_send.at<gtsam::Pose2>(timestep_to_veh_pose_key(i));
        } else if (this->impl_to_use == PoseGraphSlamImplementation::SESYNC) {
            // extract this pose from the overall combined state matrix.
            // format is [t_v1 t_v2 ... t_vn t_l1 ... t_lM | R_v1 ... R_vn R_l1 ... R_lM]
            // where each translation vector t_i is 2x1 and each rotation matrix R_i is 2x2.
            // note that all vehicle poses so far will precede the M landmarks.
            float x = this->sesync_result.xhat(0, i);
            float y = this->sesync_result.xhat(1, i);
            veh_pos = gtsam::Point2(x, y);
            // note that a rotation matrix is in the form [c, -s; s, c], so we can just
            // read the c and s values to initialize a gtsam rotation.
            float cos_theta = this->sesync_result.xhat(0, this->timestep + this->M + i);
            float sin_theta = this->sesync_result.xhat(1, this->timestep + this->M + i);
            veh_rot = gtsam::Rot2::fromCosSin(cos_theta, sin_theta);
            veh_pose_i = gtsam::Pose2(veh_rot, veh_pos);
        }
        
        x_vec.push_back(veh_pose_i.x());
        y_vec.push_back(veh_pose_i.y());
        yaw_vec.push_back(veh_pose_i.theta());
    }
    stateMsg.x_v = x_vec;
    stateMsg.y_v = y_vec;
    stateMsg.yaw_v = yaw_vec;

    // landmark position estimates.
    std::vector<float> lm_vec;
    gtsam::Vector2 lm_pos_j;
    for (int i=0; i<this->M; ++i) {
        if (this->impl_to_use == PoseGraphSlamImplementation::GTSAM) {
            int j = this->lm_IDs[i]; // Convert from landmark index to ID.
            // directly access by key.
            lm_pos_j = nodes_to_send.at<gtsam::Vector2>(landmark_id_to_key(j));
        } else if (this->impl_to_use == PoseGraphSlamImplementation::SESYNC) {
            // extract this position from the overall combined state matrix.
            // format is described above where vehicle poses are read.
            float x = this->sesync_result.xhat(0, this->timestep + i);
            float y = this->sesync_result.xhat(1, this->timestep + i);
            lm_pos_j = gtsam::Vector2(x, y);
        }
        lm_vec.push_back(lm_pos_j(0));
        lm_vec.push_back(lm_pos_j(1));
    }
    stateMsg.landmarks = lm_vec;

    // publish it for plotter.
    if (this->solved_pose_graph) {
        this->statePub.publish(stateMsg);
    } else {
        this->statePubSecondary.publish(stateMsg);
    }
}