#include "ros/ros.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <queue>
#include <iostream>
#include <string>
#include <fstream>
#include <ros/package.h>

#include "ukf_pkg/ukf.h"

// define queues for messages.
std::queue<geometry_msgs::Vector3::ConstPtr> odomQueue;
std::queue<std_msgs::Float32MultiArray::ConstPtr> lmMeasQueue;
// define state publisher.
ros::Publisher statePub;
// define UKF object from ukf.cpp class.
UKF ukf;
// config vars for params we need.
float x_0; float y_0; float yaw_0;

float readParams() {
    // read config parameters from file.
    float DT;
    std::string delimeter = " = ";
    std::string line;
    std::string pkgPath = ros::package::getPath("data_pkg");
    std::ifstream paramsFile(pkgPath+"/config/params.txt", std::ifstream::in);
    std::string token;
    // read all lines from file.
    while (getline(paramsFile, line)) {
        // only save things we need.
        token = line.substr(0, line.find(delimeter));
        if (token == "DT") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            DT = std::stof(line);
        } else if (token == "x_0") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            x_0 = std::stof(line);
        } else if (token == "y_0") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            y_0 = std::stof(line);
        } else if (token == "yaw_0") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            yaw_0 = std::stof(line);
        }
    }
    // close file.
    paramsFile.close();
    return DT;
}

void ekfIterate(const ros::TimerEvent& event) {
    // perform an iteration of the EKF for this timestep.
    if (odomQueue.empty() || lmMeasQueue.empty()) {
        return;
    }
    // get the next timestep's messages from the queues.
    geometry_msgs::Vector3::ConstPtr odomMsg = odomQueue.front();
    odomQueue.pop();
    std_msgs::Float32MultiArray::ConstPtr lmMeasMsg = lmMeasQueue.front();
    lmMeasQueue.pop();
    // call the EKF's update function.
    ukf.update(odomMsg, lmMeasMsg);
    // get the current state estimate.
    ukf_pkg::UKFState stateMsg = ukf.getState();
    // publish it.
    statePub.publish(stateMsg);
}

void odomCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    // receive an odom command and add to the queue.
    odomQueue.push(msg);
}

void lmMeasCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // receive a landmark detection and add to the queue.
    lmMeasQueue.push(msg);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ukf_node");
    ros::NodeHandle node("~");

    // read config parameters.
    float DT = readParams();
    // init the EKF.
    ukf.init(x_0, y_0, yaw_0);

    // subscribe to inputs.
    ros::Subscriber odomSub = node.subscribe("/odom", 100, odomCallback);
    ros::Subscriber lmMeasSub = node.subscribe("/landmark", 100, lmMeasCallback);
    // publish state.
    statePub = node.advertise<ukf_pkg::UKFState>("/state/ukf", 1);

    // timer to update UKF at set frequency.
    ros::Timer ekfIterationTimer = node.createTimer(ros::Duration(DT), &ekfIterate, false);

    ros::spin();
    return 0;
}