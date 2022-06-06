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
std::queue<data_pkg::Command::ConstPtr> cmdQueue;
std::queue<std_msgs::Float32MultiArray::ConstPtr> lmMeasQueue;
// define state publisher.
ros::Publisher statePub;
// define UKF object from ukf.cpp class.
UKF ukf;
// UKF mode (true for SLAM, false for localization-only).
bool ukfSlamMode;
// ukf config params.
float W_0;
// flag to wait for map to be received.
bool loadedTrueMap = false;

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
        } else if (token == "W_0") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            W_0 = std::stof(line);
        } else if (token == "ukf_mode") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            if (line == "loc") {
                ukfSlamMode = false;
            } else if (line == "slam") {
                ukfSlamMode = true;
            } else {
                std::cout << "Invalid ukf_mode in params.txt. Using SLAM mode." << std::endl << std::flush;
                ukfSlamMode = true;
            }
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
    // init the UKF.
    ukf.init(x_0, y_0, yaw_0, W_0, ukfSlamMode);
}

void ukfIterate(const ros::TimerEvent& event) {
    // perform an iteration of the UKF for this timestep.
    if (!ukf.isInit || (!loadedTrueMap && !ukfSlamMode) || cmdQueue.empty() || lmMeasQueue.empty()) {
        return;
    }
    // get the next timestep's messages from the queues.
    data_pkg::Command::ConstPtr odomMsg = cmdQueue.front();
    cmdQueue.pop();
    std_msgs::Float32MultiArray::ConstPtr lmMeasMsg = lmMeasQueue.front();
    lmMeasQueue.pop();
    // call the UKF's update function.
    ukf.ukfIterate(odomMsg, lmMeasMsg);
    // get the current state estimate.
    data_pkg::UKFState stateMsg = ukf.getState();
    // publish it.
    statePub.publish(stateMsg);
}

void cmdCallback(const data_pkg::Command::ConstPtr& msg) {
    // receive an odom command and add to the queue.
    cmdQueue.push(msg);
}

void lmMeasCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // receive a landmark detection and add to the queue.
    lmMeasQueue.push(msg);
}

void trueMapCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // set the true map to be used for localization mode.
    ukf.setTrueMap(msg);
    loadedTrueMap = true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ukf_node");
    ros::NodeHandle node("~");

    // read config parameters.
    float DT = readParams();
    // get the initial veh pose and init the ukf.
    ros::Subscriber initSub = node.subscribe("/truth/init_veh_pose", 1, initCallback);

    // subscribe to inputs.
    ros::Subscriber cmdSub = node.subscribe("/command", 100, cmdCallback);
    ros::Subscriber lmMeasSub = node.subscribe("/landmark", 100, lmMeasCallback);
    ros::Subscriber trueMapSub = node.subscribe("/truth/landmarks", 1, trueMapCallback);
    // publish state.
    statePub = node.advertise<data_pkg::UKFState>("/state/ukf", 1);

    // timer to update UKF at set frequency.
    ros::Timer ekfIterationTimer = node.createTimer(ros::Duration(DT), &ukfIterate, false);

    ros::spin();
    return 0;
}