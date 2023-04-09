#include "localization_pkg/filter.h"

// Some of this implementation is based on an example from GTSAM itself:
// https://github.com/borglab/gtsam/blob/develop/examples/Pose2SLAMExample.cpp

// init the PoseGraph.
PoseGraph::PoseGraph() {
    // set filter type.
    this->type = FilterChoice::POSE_GRAPH_SLAM;
    ///\todo: initialization and whatnot.
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

    // define noise models for both types of connections.
    this->process_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(this->V(0,0), this->V(0,0), this->V(1,1)));
    this->sensing_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(this->W(0,0), this->W(0,0), this->W(1,1)));
}

void PoseGraph::init(float x_0, float y_0, float yaw_0) {
    // Add a prior on the first pose, setting it to the origin
    // A prior factor consists of a mean and a noise model (covariance matrix)
    auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(0.3, 0.3, 0.1));
    ROS_INFO_STREAM("PGS: Adding prior for initial veh pose.");
    graph.addPrior(this->timestep, gtsam::Pose2(x_0, y_0, yaw_0), priorNoise);

    ROS_INFO_STREAM("PGS: Adding prior as first naive estimate.");
    this->initial_estimate.insert(this->timestep, gtsam::Pose2(x_0, y_0, yaw_0));

    // set initialized flag.
    this->isInit = true;
}

void PoseGraph::updateNaiveVehPoseEstimate(float x, float y, float yaw) {
    // Update our current naive belief of the vehicle pose.
    ///\note: This estimate may come from another filter such as the EKF, or could be a basic propagation with no filtering.
    this->x_t.setZero(3);
    this->x_t << x, y, yaw;

    if (this->timestep <= this->graph_size_threshold) {
        // Add to the estimate list for the pose-graph optimization starting iterate.
        ROS_INFO_STREAM("PGS: Adding naive estimate.");
        this->initial_estimate.insert(this->timestep+1, gtsam::Pose2(x, y, yaw));
    } // else, we already ran optimization, so just exit.
}

void PoseGraph::onLandmarkMeasurement(int id, float range, float bearing) {
    // convert this measurement into a transformation matrix from the current vehicle pose.
    Eigen::MatrixXd meas_mat = Filter::computeTransform(range, bearing);
    // determine measured location of landmark according to this and most recent vehicle pose.
    Eigen::MatrixXd cur_veh_pose;
    cur_veh_pose.setIdentity(3,3);
    cur_veh_pose.block(0,0,2,2) = Filter::yawToMat(this->x_t(2));
    cur_veh_pose(0,2) = this->x_t(0);
    cur_veh_pose(1,2) = this->x_t(1);
    Eigen::MatrixXd meas_lm_pose = meas_mat * cur_veh_pose;
    // convert this pose to a simple x,y position, since landmarks have no discernable orientation.
    Eigen::Vector2d meas_landmark_pos(meas_lm_pose(0,2), meas_lm_pose(1,2));

    int landmark_meas_index = -1;
    if (this->landmark_id_is_known) {
        // Check if the landmark with this ID has been detected before.
        for (int i=0; i < this->M; ++i) {
            if (this->lm_IDs[i] == id) {
                // We have detected this landmark before, so note the index.
                landmark_meas_index = i;
                break;
            }
        }
    } else {
        // the ID is not given, so check all landmarks to see if this is close to one we've seen before.
        for (int l=0; l<this->M; ++l) {
            int lm_id = this->lm_IDs[l];
            if (Filter::vecDistance(meas_landmark_pos, this->lm_positions[lm_id]) < this->min_landmark_separation) {
                // Declare that we have detected this landmark before, so note the index.
                landmark_meas_index = lm_id;
                break;
            }
        }
    }

    if (landmark_meas_index == -1) {
        // This is the first detection of this landmark.
        landmark_meas_index = this->M;
        if (this->landmark_id_is_known) {
            this->lm_IDs.push_back(id);
        } else {
            this->lm_IDs.push_back(landmark_meas_index);
        }
        this->M++;
        this->lm_positions.push_back(meas_landmark_pos);
        // Create the vector that will store all measurement transforms.
        std::unordered_map<int, Eigen::MatrixXd> meas_vec;
        this->measurements.push_back(meas_vec);
    }
    
    // save this measured landmark position.
    ///\todo: maybe update our estimate of this landmark's position somehow (if this isn't the first detection), rather than simply overwriting it?
    this->lm_positions[landmark_meas_index] = meas_landmark_pos;

    ///\todo: if we've detected a previously-seen landmark, generate a loop-closure constraint with every previous vehicle pose that observed this landmark.
    Eigen::MatrixXd inv_meas_mat = meas_mat.inverse();
    for (auto& meas_map: this->measurements[landmark_meas_index]) {
        int iteration_number = meas_map.first;
        // Combine pair of measurement transforms to generate a single relationship for a graph connection.
        Eigen::MatrixXd connection_mat = inv_meas_mat * this->measurements[landmark_meas_index][iteration_number];
        // Create the loop closure connection.
        ROS_INFO_STREAM("PGS: Adding loop closure connection.");
        this->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2> >(iteration_number, this->timestep, gtsam::Pose2(connection_mat), sensing_noise_model);
    }

    // Add this new measurement to the map for this landmark.
    this->measurements[landmark_meas_index][this->timestep] = meas_mat;
}

// Update the graph with the new information for this iteration.
void PoseGraph::update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // update timestep (i.e., iteration index).
    this->timestep += 1;

    // check stopping criteria.
    if (this->timestep == this->graph_size_threshold) {
        // run optimization algorithm and then exit.
        ROS_INFO_STREAM("PGS: Solving the pose graph.");
        solvePoseGraph();
        return;
    } else if (this->timestep > this->graph_size_threshold) {
        // we already ran optimization, so just exit.
        // ROS_WARN_STREAM("PGS: Already solved pose graph, so just exiting immediately.");
        return;
    }

    // use commanded motion to create a connection between previous and new vehicle pose.
    ROS_INFO_STREAM("PGS: Adding command connection.");
    this->graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose2> >(this->timestep-1, this->timestep, gtsam::Pose2(cmdMsg->fwd, 0, cmdMsg->ang), process_noise_model);

    // process all measurements for this iteration, and generate loop closure constraints if applicable.
    std::vector<float> lm_meas = lmMeasMsg->data;
    int num_landmarks = (int) (lm_meas.size() / 3);
    // if there is at least one detection, handle each individually.
    for (int l=0; l<num_landmarks; ++l) {
        // extract the landmark details. only use the ID if we're allowed to. otherwise, do data association.
        int id = -1;
        if (this->landmark_id_is_known) {
            int id = (int) lm_meas[l*3];
        }
        float r = lm_meas[l*3+1];
        float b = lm_meas[l*3+2];
        onLandmarkMeasurement(id, r, b);
    }
}

void PoseGraph::solvePoseGraph() {
    // print the pose graph and our initial estimate of all vehicle poses.
    this->graph.print("\nFactor Graph:\n");
    this->initial_estimate.print("\nInitial Estimate:\n");

    // Optimize the initial values using a Gauss-Newton nonlinear optimizer
    // The optimizer accepts an optional set of configuration parameters.
    gtsam::GaussNewtonParams parameters;
    parameters.relativeErrorTol = this->iteration_error_threshold;
    parameters.maxIterations = this->max_iterations;
    // Create the optimizer ...
    gtsam::GaussNewtonOptimizer optimizer(this->graph, this->initial_estimate, parameters);
    // ... and optimize
    gtsam::Values result = optimizer.optimize();
    result.print("Final Result:\n");

    // Calculate and print marginal covariances for all variables
    std::cout.precision(3);
    gtsam::Marginals marginals(graph, result);
    for (int i = 0; i < this->timestep; ++i) {
        std::cout << "covariance of vehicle pose " << i << ":\n" << marginals.marginalCovariance(i) << std::endl;
    }

    ///\todo: Save or display the results. Compare to naive estimates analytically and visually.
}
