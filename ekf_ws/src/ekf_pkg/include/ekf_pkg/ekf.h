#ifndef IGVC_EKF_H
#define IGVC_EKF_H

#include <ros/ros.h>
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include "ekf_pkg/EKFState.h"
#include "data_pkg/Command.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>

#define pi 3.14159265358979323846

class EKF {
    public:
    // current timestep.
    int timestep = 0;
    // state distribution.
    Eigen::VectorXd x_t;
    Eigen::MatrixXd P_t;
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

    EKF();
    void init(float x_0, float y_0, float yaw_0);
    void update(data_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg);
    ekf_pkg::EKFState getState();
};

#endif