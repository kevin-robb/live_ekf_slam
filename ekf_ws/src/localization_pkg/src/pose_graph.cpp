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
    Filter::readParams(config);
    // setup all filter-specific params, if any.
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
    graph.addPrior(this->timestep, gtsam::Pose2(x_0, y_0, yaw_0), priorNoise);

    // set initialized flag.
    this->isInit = true;
}

Eigen::Matrix2d PoseGraph::yawToMat(float theta) {
    Eigen::Matrix2d result;
    result.setIdentity(2,2);
    result(0,0) = cos(theta);
    result(0,1) = -sin(theta);
    result(1,0) = sin(theta);
    result(1,1) = cos(theta);
    return result;
}

float PoseGraph::matToYaw(Eigen::Matrix2d R_theta) {
    return atan2(R_theta(1,0), R_theta(0,0));
}

Eigen::Matrix3d PoseGraph::computeTransform(float fwd, float ang) {
    Eigen::Matrix3d transform_mat;
    transform_mat.setIdentity(3,3);
    transform_mat.block(0,0,2,2) = yawToMat(ang);
    transform_mat(0,2) = fwd;
    return transform_mat;
}

float PoseGraph::vecDistance(Eigen::Vector2d v1, Eigen::Vector2d v2) {
    // compute euclidean distance between the two position vectors.
    return sqrt((v1(0)-v2(0))*(v1(0)-v2(0)) + (v1(1)-v2(1))*(v1(1)-v2(1)));
}

void PoseGraph::onLandmarkMeasurement(int id, float range, float bearing) {
    // convert this measurement into a transformation matrix from the current vehicle pose.
    Eigen::Matrix3d meas_mat = computeTransform(range, bearing);
    // determine measured location of landmark according to this and most recent vehicle pose.
    // Eigen::Matrix3d meas_lm_pose = meas_mat * this->vehicle_poses.back();
    // convert this pose to a simple x,y position, since landmarks have no discernable orientation.
    // Eigen::Vector2d meas_landmark_pos(meas_lm_pose(0,2), meas_lm_pose(1,2));

    // int landmark_id = -1;
    // if (this->landmark_id_is_known) {
    //     landmark_id = id;
    // } else {
    //     // the ID is not given, so check all landmarks to see if this is close to one we've seen before.
    //     for (int l=0; l<this->M; ++l) {
    //         int lm_id = this->lm_IDs[l];
    //         if (vecDistance(meas_landmark_pos, this->landmark_positions.at(lm_id)) < this->min_landmark_separation) {
    //             // declare these are the same landmark.
    //             landmark_id = lm_id;
    //             break;
    //         }
    //     }
    //     if (landmark_id == -1) {
    //         // this is the first time detecting this landmark, so add its position to our list of landmark nodes.
    //         landmark_id = this->M;
    //         this->lm_IDs.push_back(landmark_id);
    //         this->M++;
    //         this->landmark_positions.push_back(meas_landmark_pos);
    //     }
    // }
    // // save this measured landmark position.
    // ///\todo: maybe update our estimate of this landmark's position somehow (if this isn't the first detection), rather than simply overwriting it?
    // this->landmark_positions[landmark_id] = meas_landmark_pos;

    // // add the relative transform matrix as a connection.
    // this->measurements[make_pair(this->timestep,landmark_id)] = meas_mat;
}

// Update the graph with the new information for this iteration.
void PoseGraph::update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // update timestep (i.e., iteration index).
    this->timestep += 1;

    // use commanded motion to create a connection between previous and new vehicle pose.
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
        ///\todo: if we've detected a previously-seen landmark, generate a loop-closure constraint with every previous vehicle pose that observed this landmark.
    }

    ///\todo: Add "initial estimate" poses to a graph. These will come from running the EKF, UKF, or just doing some dumb propagation method.
    float x_est, y_est, yaw_est;
    this->initial_estimate.insert(this->timestep, gtsam::Pose2(x_est, y_est, yaw_est));

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
}
