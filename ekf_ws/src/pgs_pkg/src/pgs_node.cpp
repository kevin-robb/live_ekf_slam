
#include <ros/ros.h>
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include "data_pkg/PGSState.h"
#include "data_pkg/Command.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>
#include <queue>
#include <iostream>
#include <string>
#include <fstream>
#include <ros/package.h>

#include <g2o/core>


// define queues for messages.
std::queue<data_pkg::Command::ConstPtr> cmdQueue;
std::queue<std_msgs::Float32MultiArray::ConstPtr> lmMeasQueue;
// define EKF state publisher.
ros::Publisher statePub;

float readParams() {
    // read config parameters from file.
    float DT;
    std::string delimeter = " = ";
    std::string line;
    // std::string pkgPath = ros::package::getPath("data_pkg");
    std::string pkgPath = "/home/kevin-robb/Projects/live_ekf_slam/ekf_ws/src/data_pkg";
    std::ifstream paramsFile(pkgPath+"/config/params.txt", std::ifstream::in);
    std::string token;
    // read all lines from file.
    while (getline(paramsFile, line)) {
        // only save things we need.
        token = line.substr(0, line.find(delimeter));
        if (token == "DT") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            DT = std::stof(line);
        }
    }
    // close file.
    paramsFile.close();
    return DT;
}

void initCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    // receive the vehicle's initial position.
    float x_0 = msg->x;
    float y_0 = msg->y;
    float yaw_0 = msg->z;
    // init the EKF.
    // ekf.init(x_0, y_0, yaw_0);
}

void ekfIterate(const ros::TimerEvent& event) {
    // perform an iteration of the EKF for this timestep.
    if (cmdQueue.empty() || lmMeasQueue.empty()) {
        return;
    }
    // get the next timestep's messages from the queues.
    data_pkg::Command::ConstPtr cmdMsg = cmdQueue.front();
    cmdQueue.pop();
    std_msgs::Float32MultiArray::ConstPtr lmMeasMsg = lmMeasQueue.front();
    lmMeasQueue.pop();
    // call the EKF's update function.
    // ekf.update(cmdMsg, lmMeasMsg);
    // get the current state estimate.
    // data_pkg::EKFState stateMsg = ekf.getState();
    // publish it.
    // statePub.publish(stateMsg);
}

void cmdCallback(const data_pkg::Command::ConstPtr& msg) {
    // receive an odom command and add to the queue.
    cmdQueue.push(msg);
}

void lmMeasCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // receive a landmark detection and add to the queue.
    lmMeasQueue.push(msg);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ekf_node");
    ros::NodeHandle node("~");

    // read config parameters.
    float DT = readParams();
    // get the initial veh pose and init the ekf.
    ros::Subscriber initSub = node.subscribe("/truth/init_veh_pose", 1, initCallback);

    // subscribe to EKF inputs.
    ros::Subscriber cmdSub = node.subscribe("/command", 100, cmdCallback);
    ros::Subscriber lmMeasSub = node.subscribe("/landmark", 100, lmMeasCallback);
    // publish EKF state.
    // statePub = node.advertise<data_pkg::EKFState>("/state/ekf", 1);

    // timer to update EKF at set frequency.
    ros::Timer ekfIterationTimer = node.createTimer(ros::Duration(DT), &ekfIterate, false);

    ros::spin();
    return 0;
}