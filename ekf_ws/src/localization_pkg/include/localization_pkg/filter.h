#ifndef FILTER_INTERFACE_H
#define FILTER_INTERFACE_H

// macro to make eigen use exceptions instead of assertion fails.
#define eigen_assert(X) do { if(!(X)) throw std::runtime_error(#X); } while(false);

// Standard imports.
#include <ros/ros.h>
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <vector>
#include <cmath>
#include <typeinfo>
#include <eigen3/Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <eigen3/Eigen/Eigenvalues>
#include <yaml-cpp/yaml.h>

// Custom imports.
#include "base_pkg/Command.h"
#include "base_pkg/EKFState.h"
#include "base_pkg/UKFState.h"

#define pi 3.14159265358979323846

enum class FilterChoice {
    EKF_SLAM,
    UKF_LOC,
    UKF_SLAM,
    POSE_GRAPH_SLAM
};

// Interface class for stuff used by all filters.
class Filter {
public:
    FilterChoice type;
    Filter() {}; // Do nothing.
    virtual ~Filter() {}; // Do nothing.
    virtual void init(float x_0, float y_0, float yaw_0) = 0;
    virtual void update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) = 0;
    // State calls needed to avoid errors :/
    base_pkg::EKFState getEKFState() { throw std::runtime_error("Cannot call getEKFState for this filter."); base_pkg::EKFState s; return s; };
    base_pkg::UKFState getUKFState() { throw std::runtime_error("Cannot call getUKFState for this filter."); base_pkg::UKFState s; return s; };

    bool isInit = false;
    // true set of landmark positions for localization-only filters/modes.
    std::vector<float> map;

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
    int timestep = 0; // current timestep.
    int M = 0; // number of landmarks being tracked.
    std::vector<int> lm_IDs; // IDs of landmarks being tracked.
    // state distribution.
    Eigen::VectorXd x_t;
    Eigen::MatrixXd P_t;
    // predicted state distribution.
    Eigen::VectorXd x_pred;
    Eigen::MatrixXd P_pred;

public:
    void readParams(YAML::Node config) { // Read/setup commonly-used params.
        // process noise.
        this->V.setIdentity(2,2);
        this->v_d = config["process_noise"]["mean"]["v_d"].as<float>();
        this->v_th = config["process_noise"]["mean"]["v_th"].as<float>();
        this->V(0,0) = config["process_noise"]["cov"]["V_00"].as<float>();
        this->V(1,1) = config["process_noise"]["cov"]["V_11"].as<float>();
        // sensing noise.
        this->W.setIdentity(2,2);
        this->w_r = config["sensing_noise"]["mean"]["w_r"].as<float>();
        this->w_b = config["sensing_noise"]["mean"]["w_b"].as<float>();
        this->V(0,0) = config["sensing_noise"]["cov"]["W_00"].as<float>();
        this->V(1,1) = config["sensing_noise"]["cov"]["W_11"].as<float>();
        // measurement constraints.
        this->landmark_id_is_known = config["constraints"]["measurements"]["landmark_id_is_known"].as<bool>();
        this->min_landmark_separation = config["constraints"]["measurements"]["min_landmark_separation"].as<float>();
    }
};

// Extended Kalman Filter for SLAM.
class EKF: public Filter {
public:
    FilterChoice type = FilterChoice::EKF_SLAM;
    EKF();
    ~EKF() {}; // Do nothing.
    void readParams(YAML::Node config);
    void init(float x_0, float y_0, float yaw_0);
    void update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    base_pkg::EKFState getEKFState();

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
    FilterChoice type = FilterChoice::UKF_SLAM;
    UKF();
    ~UKF() {}; // Do nothing.
    void readParams(YAML::Node config);
    void init(float x_0, float y_0, float yaw_0);
    void update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    base_pkg::UKFState getUKFState();
    Eigen::MatrixXd nearestSPD();
    Eigen::VectorXd motionModel(Eigen::VectorXd x, float u_d, float u_th);
    Eigen::VectorXd sensingModel(Eigen::VectorXd x, int lm_i);
    void predictionStage(base_pkg::Command::ConstPtr cmdMsg);
    void updateStage(std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    void landmarkUpdate(int lm_i, int id, float r, float b);
    void landmarkInsertion(int id, float r, float b);

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
    FilterChoice type = FilterChoice::POSE_GRAPH_SLAM;
    PoseGraph();
    ~PoseGraph() {}; // Do nothing.
    void readParams(YAML::Node config);
    void init(float x_0, float y_0, float yaw_0);
    void update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);

protected:

};

#endif // FILTER_INTERFACE_H