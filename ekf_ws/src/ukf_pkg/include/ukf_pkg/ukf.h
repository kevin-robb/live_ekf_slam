#ifndef UKF_H
#define UKF_H

// macro to make eigen use exceptions instead of assertion fails.
#define eigen_assert(X) do { if(!(X)) throw std::runtime_error(#X); } while(false);

#include <ros/ros.h>
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include "data_pkg/UKFState.h"
#include "data_pkg/Command.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>
#include <eigen3/Eigen/Eigenvalues>

#define pi 3.14159265358979323846

class UKF {
    public:
    bool isInit = false;
    bool ukfSlamMode;
    // current timestep.
    int timestep = 0;
    // state distribution.
    Eigen::VectorXd x_t;
    Eigen::MatrixXd P_t;
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
    // predicted state distribution.
    Eigen::VectorXd x_pred;
    Eigen::Vector2d complex_angle =  Eigen::Vector2d::Zero(2); // temp storage for converting hdg to complex # during averaging.
    Eigen::MatrixXd P_pred;
    Eigen::MatrixXd p_temp; // temp matrix used for P update.
    // landmark IDs.
    std::vector<int>  lm_IDs;
    // number of landmarks in the state.
    int M = 0;
    // process noise.
    float v_d = 0;
    float v_th = 0;
    Eigen::MatrixXd V;
    Eigen::MatrixXd Q; // will expand for cov summation.
    // sensing noise.
    float w_r = 0;
    float w_b = 0;
    Eigen::MatrixXd W;
    // true map for localization-only mode.
    std::vector<float> map;

    UKF();
    void init(float x_0, float y_0, float yaw_0);
    data_pkg::UKFState getState();
    Eigen::MatrixXd nearestSPD();
    Eigen::VectorXd motionModel(Eigen::VectorXd x, float u_d, float u_th);
    Eigen::VectorXd sensingModel(Eigen::VectorXd x, int lm_i);
    void ukfIterate(data_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    void predictionStage(data_pkg::Command::ConstPtr cmdMsg);
    void updateStage(std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    void landmarkUpdate(int lm_i, int id, float r, float b);
    void landmarkInsertion(int id, float r, float b);
};

#endif