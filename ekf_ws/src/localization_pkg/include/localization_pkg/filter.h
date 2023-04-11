#ifndef FILTER_INTERFACE_H
#define FILTER_INTERFACE_H

// macro to make eigen use exceptions instead of assertion fails.
#define eigen_assert(X) do { if(!(X)) throw std::runtime_error(#X); } while(false);

// Standard C++ and ROS imports.
#include <ros/ros.h>
#include <ros/console.h>
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <vector>
#include <cmath>
#include <typeinfo>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
// Eigen.
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <eigen3/Eigen/Eigenvalues>
// GTSAM.
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Key.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <memory>
// Custom message type imports.
#include "base_pkg/Command.h"
#include "base_pkg/EKFState.h"
#include "base_pkg/UKFState.h"
#include "base_pkg/PoseGraphState.h"

#define pi 3.14159265358979323846

enum class FilterChoice {
    NOT_SET = 0,
    EKF_SLAM,
    UKF_LOC,
    UKF_SLAM,
    POSE_GRAPH_SLAM,
    NAIVE_COMMAND_PROPAGATION // No filtering will be done, but rather commands will be directly applied to the vehicle pose estimate. This can be used as a baseline for the initial pose graph iterate.
};

// Interface class for stuff used by all filters.
class Filter {
public:
    FilterChoice type = FilterChoice::NOT_SET;
    Filter() {}; // Do nothing.
    virtual ~Filter() {}; // Do nothing.
    virtual void readParams(YAML::Node config) = 0;
    virtual void init(float x_0, float y_0, float yaw_0) = 0;
    virtual void update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) = 0;

    // Publish the current filter state to the proper ROS topic & message type.
    ros::Publisher statePub; // Will be defined as the correct type by the individual filter.
    virtual void setupStatePublisher(ros::NodeHandle node) = 0;
    virtual void publishState() = 0;

    bool isInit = false;
    // true set of landmark positions for localization-only filters/modes.
    std::vector<float> map;

    // Some filters may want a second filter to run in tandem for comparison. Must define here to avoid errors in localization_node.
    FilterChoice filter_to_compare = FilterChoice::NOT_SET; // Filter to run simultaneously as naive estimate.
    virtual void updateNaiveVehPoseEstimate(float x, float y, float yaw) { throw std::runtime_error("updateNaiveVehPoseEstimate is not defined for this filter."); };
    // Similarly, some filters may not be able to be run as a secondary filter, since they don't run online.
    virtual Eigen::Vector3d getStateVector() { throw std::runtime_error("getStateVector is not defined for this filter."); };


protected:
    // Standard configs/flags.
    bool landmark_id_is_known = false;
    float min_landmark_separation; // two detections less than this distance apart will be considered the same landmark.

    // process noise.
    float v_d = 0;
    float v_th = 0;
    Eigen::MatrixXd V;
    // sensing noise.
    float w_r = 0;
    float w_b = 0;
    Eigen::MatrixXd W;

    // State tracking.
    int timestep = 0; // current timestep (i.e., iteration index).
    int M = 0; // number of landmarks being tracked (so far).
    std::vector<int> lm_IDs; // IDs of landmarks being tracked. length should be M.
    // state distribution.
    Eigen::VectorXd x_t;
    Eigen::MatrixXd P_t;
    // predicted state distribution.
    Eigen::VectorXd x_pred;
    Eigen::MatrixXd P_pred;

public:
    // Implementations of functions that are common among muiltiple filters.
    void readCommonParams(YAML::Node config) { // Read/setup commonly-used params.
        // process noise.
        this->V.setIdentity(2,2);
        this->v_d = config["process_noise"]["mean"]["v_d"].as<float>();
        this->v_th = config["process_noise"]["mean"]["v_th"].as<float>();
        this->V(0,0) = config["process_noise"]["cov"]["V_00"].as<double>();
        this->V(1,1) = config["process_noise"]["cov"]["V_11"].as<double>();
        // sensing noise.
        this->W.setIdentity(2,2);
        this->w_r = config["sensing_noise"]["mean"]["w_r"].as<float>();
        this->w_b = config["sensing_noise"]["mean"]["w_b"].as<float>();
        this->V(0,0) = config["sensing_noise"]["cov"]["W_00"].as<double>();
        this->V(1,1) = config["sensing_noise"]["cov"]["W_11"].as<double>();
        // measurement constraints.
        this->landmark_id_is_known = config["constraints"]["measurements"]["landmark_id_is_known"].as<bool>();
        this->min_landmark_separation = config["constraints"]["measurements"]["min_landmark_separation"].as<float>();
    }
    Eigen::Matrix2d yawToMat(float theta) {
        Eigen::Matrix2d result;
        result.setIdentity(2,2);
        result(0,0) = cos(theta);
        result(0,1) = -sin(theta);
        result(1,0) = sin(theta);
        result(1,1) = cos(theta);
        return result;
    }
    float matToYaw(Eigen::Matrix2d R_theta) {
        return atan2(R_theta(1,0), R_theta(0,0));
    }
    Eigen::MatrixXd computeTransform(float fwd, float ang) {
        Eigen::MatrixXd transform_mat;
        transform_mat.setIdentity(3,3);
        transform_mat.block(0,0,2,2) = yawToMat(ang);
        transform_mat(0,2) = fwd;
        return transform_mat;
    }
    float vecDistance(Eigen::Vector3d v1, Eigen::Vector3d v2) {
        // compute euclidean distance between the two position vectors.
        return sqrt((v1(0)-v2(0))*(v1(0)-v2(0)) + (v1(1)-v2(1))*(v1(1)-v2(1)));
    }
};

// Extended Kalman Filter for SLAM.
class EKF: public Filter {
public:
    EKF();
    ~EKF() {}; // Do nothing.
    void readParams(YAML::Node config);
    void init(float x_0, float y_0, float yaw_0);
    void update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);

    Eigen::Vector3d getStateVector();
    void setupStatePublisher(ros::NodeHandle node);
    void publishState();

protected:
    // jacobians and such.
    Eigen::MatrixXd F_x; //prediction jacobians
    Eigen::MatrixXd F_v;
    Eigen::MatrixXd H_x; //update jacobians
    Eigen::MatrixXd H_w;
    Eigen::Vector2d nu; //innovation
    Eigen::MatrixXd S; //innovation cov
    Eigen::MatrixXd K; //kalman gain
    Eigen::MatrixXd G_z;
    Eigen::MatrixXd G_x;
    Eigen::MatrixXd Y; //insertion jacobian
    // intermediary matrix used to update cov.
    Eigen::MatrixXd p_temp;
};

// Unscented Kalman Filter for localization-only or SLAM.
class UKF: public Filter {
public:
    UKF();
    ~UKF() {}; // Do nothing.
    void readParams(YAML::Node config);
    void init(float x_0, float y_0, float yaw_0);
    void update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    Eigen::MatrixXd nearestSPD();
    Eigen::VectorXd motionModel(Eigen::VectorXd x, float u_d, float u_th);
    Eigen::VectorXd sensingModel(Eigen::VectorXd x, int lm_i);
    void predictionStage(base_pkg::Command::ConstPtr cmdMsg);
    void updateStage(std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    void landmarkUpdate(int lm_i, int id, float r, float b);
    void landmarkInsertion(int id, float r, float b);

    Eigen::Vector3d getStateVector();
    void setupStatePublisher(ros::NodeHandle node);
    void publishState();

protected:
    Eigen::MatrixXd Q; // expanding process noise matrix for cov summation.
    // stuff used for P_t approx SPD calculation.
    Eigen::MatrixXd Y;
    Eigen::VectorXd D;
    Eigen::MatrixXd Qv;
    Eigen::VectorXd Dplus;
    Eigen::MatrixXd sqtP; // sqrt of P_t term for sigma pts calc.
    // sigma points matrix and weights.
    Eigen::MatrixXd X; // sigma points.
    Eigen::VectorXd Wts; // weights for sigma pts.
    float W_0 = 0.2; // mean weight.
    Eigen::MatrixXd X_pred; // propagated sigma pts.
    Eigen::MatrixXd X_zest; // meas ests for each sigma pt.
    Eigen::VectorXd diff; // meas est differences for computing covariances.
    Eigen::VectorXd diff2; // meas est differences for computing covariances.
    Eigen::VectorXd z_est; // combined z_est.
    Eigen::MatrixXd C; // cross cov b/w x_pred and z_est.
    Eigen::MatrixXd S; // innov cov.
    Eigen::MatrixXd K; // kalman gain.

    // current measurement.
    Eigen::VectorXd z = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd innovation = Eigen::VectorXd::Zero(2);

    Eigen::Vector2d complex_angle =  Eigen::Vector2d::Zero(2); // temp storage for converting hdg to complex # during averaging.
    Eigen::MatrixXd p_temp; // temp matrix used for P update.
};

// Pose-Graph optimization for SLAM.
class PoseGraph: public Filter {
public:
    PoseGraph();
    ~PoseGraph() {}; // Do nothing.
    void readParams(YAML::Node config);
    void init(float x_0, float y_0, float yaw_0);
    void update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    // PG-SLAM-specific functions.
    void onLandmarkMeasurement(int id, float range, float bearing); // Process a single landmark measurement.
    void solvePoseGraph();

    void publishState();

protected:
    // The pose graph that will be optimized to solve for full vehicle history.
    gtsam::NonlinearFactorGraph graph;
    // Noise models for connections.
    std::shared_ptr<gtsam::noiseModel::Diagonal> process_noise_model;
    std::shared_ptr<gtsam::noiseModel::Diagonal> sensing_noise_model;

    // Estimates of all poses & landmarks before running pose-graph optimization.
    // This is the initial iterate used when running the algorithm.
    gtsam::Values initial_estimate;
    // Estimated full pose history & landmarks after running PG optimization.
    gtsam::Values result;

    // Stopping criteria.
    int graph_size_threshold;
    bool solved_pose_graph = false; // Don't publish pose graph until it's been solved.

    // Params for optimization algorithm.
    float iteration_error_threshold;
    int max_iterations;

    // Desired logging behavior.
    bool verbose = false;

    // Current estimated vehicle pose from the naive filter.
    // This is used for determining landmark measurements.
    gtsam::Pose2 cur_veh_pose_estimate;

    // Stuff to help with pgs-specific message publishing.
    // We want to publish both the initial pose graph from the naive filter's estimates as well as the optimized graph, so we need two publishers.
    ros::Publisher statePubSecondary;
    // Encoded graph connections between vehicle poses and landmarks. This is just used for drawing lines in the visualization.
    // list of connections takes the form [i_1,j_1,...,i_k,j_k] where i_t is an iteration number representing a vehicle pose, and j_m is a particular landmark index.
    std::vector<int> msg_measurement_connections; 

    //////// Basic helper functions //////////////
    uint64_t timestep_to_veh_pose_key(int timestep) {
        if (timestep < 0) { // Invalid.
            throw std::runtime_error("Called timestep_to_veh_pose_key() with invalid timestep.");
        }
        // Create a key that does not conflict with landmark keys.
        return (uint64_t)timestep * 2;
    }

    uint64_t landmark_id_to_key(int landmark_id) {
        if (landmark_id < 0) { // Invalid.
            throw std::runtime_error("Called landmark_id_to_key() with invalid landmark_id.");
        }
        // Create a key for the landmark that does not conflict with any vehicle poses.
        return (uint64_t)landmark_id * 2 + 1;
    }

public:
    void setupStatePublisher(ros::NodeHandle node) {
        // Create a publisher for the proper state message type.
        this->statePubSecondary = node.advertise<base_pkg::PoseGraphState>("/state/pose_graph/initial", 1);
        this->statePub = node.advertise<base_pkg::PoseGraphState>("/state/pose_graph/result", 1);
    }

    // The localization_node will handle running a second filter and letting us know its estimates.
    void updateNaiveVehPoseEstimate(float x, float y, float yaw) {
        // Update our current naive belief of the vehicle pose.
        ///\note: This estimate may come from another filter such as the EKF, or could be a basic propagation with no filtering.
        // Save this estimate directly as a pose matrix.
        this->cur_veh_pose_estimate = gtsam::Pose2(x, y, yaw);
        // This will be added as a node in the pose graph during the update() loop.
    }
};

#endif // FILTER_INTERFACE_H