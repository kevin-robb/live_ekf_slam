#ifndef IGVC_UKF_H
#define IGVC_UKF_H

#include <ros/ros.h>
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include "ukf_pkg/UKFState.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
// #include <unsupported/Eigen/MatrixFunctions>


#define pi 3.14159265358979323846

class UKF {
    public:
    // current timestep.
    int timestep = 0;
    // state distribution.
    Eigen::VectorXd x_t;
    Eigen::MatrixXd P_t;
    Eigen::MatrixXd sqtP; // sqrt of P_t for sigma pts calc.
    // sigma points matrix and weights.
    Eigen::MatrixXd X; // sigma points.
    Eigen::VectorXd Wts; // weights for sigma pts.
    float W_0 = 0.2; // mean weight.
    Eigen::MatrixXd X_pred; // propagated sigma pts.
    Eigen::MatrixXd X_zest; // meas ests for each sigma pt.
    Eigen::VectorXd z_est; // combined z_est.
    Eigen::MatrixXd C; // cross cov b/w x_pred and z_est.
    // current measurement.
    Eigen::VectorXd z = Eigen::VectorXd::Zero(2);
    // predicted state distribution.
    Eigen::VectorXd x_pred;
    Eigen::MatrixXd P_pred;
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

    UKF();
    void update(geometry_msgs::Vector3::ConstPtr odomMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    Eigen::VectorXd motionModel(Eigen::VectorXd x, float u_d, float u_th);
    Eigen::VectorXd sensingModel(Eigen::VectorXd x, int lm_i);
    ukf_pkg::UKFState getState();
};

#endif