#include "ros/ros.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <queue>
#include <iostream>
#include <string>
#include <fstream>
#include <ros/package.h>
// #include <boost/algorithm/string>

#include "ukf_pkg/ukf.h"

// define queues for messages.
std::queue<base_pkg::Command::ConstPtr> cmdQueue;
std::queue<std_msgs::Float32MultiArray::ConstPtr> lmMeasQueue;
// define state publisher.
ros::Publisher statePub;
// define UKF object from ukf.cpp class.
UKF ukf;
// flag to wait for map to be received.
bool loadedTrueMap = false;

float readParams() {
    // read config parameters from file.
    float DT;
    std::string delimeter = " = ";
    std::string line;
    std::string pkgPath = ros::package::getPath("base_pkg");
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
            ukf.W_0 = std::stof(line);
        } else if (token == "v_d") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            ukf.v_d = std::stof(line);
        } else if (token == "v_th") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            ukf.v_th = std::stof(line);
        } else if (token == "w_r") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            ukf.w_r = std::stof(line);
        } else if (token == "w_b") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            ukf.w_b = std::stof(line);
        } else if (token == "V_00") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            ukf.V(0,0) = std::stof(line);
        } else if (token == "V_11") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            ukf.V(1,1) = std::stof(line);
        } else if (token == "W_00") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            ukf.W(0,0) = std::stof(line);
        } else if (token == "W_11") {
            line.erase(0, line.find(delimeter)+delimeter.length());
            ukf.W(1,1) = std::stof(line);
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
    ukf.init(x_0, y_0, yaw_0);
}

void ukfIterate(const ros::TimerEvent& event) {
    // perform an iteration of the UKF for this timestep.
    if (!ukf.isInit || (!loadedTrueMap && !ukf.ukfSlamMode) || cmdQueue.empty() || lmMeasQueue.empty()) {
        return;
    }
    // get the next timestep's messages from the queues.
    base_pkg::Command::ConstPtr odomMsg = cmdQueue.front();
    cmdQueue.pop();
    std_msgs::Float32MultiArray::ConstPtr lmMeasMsg = lmMeasQueue.front();
    lmMeasQueue.pop();
    // call the UKF's update function.
    ukf.ukfIterate(odomMsg, lmMeasMsg);
    // get the current state estimate.
    base_pkg::UKFState stateMsg = ukf.getState();
    // publish it.
    statePub.publish(stateMsg);
}

void cmdCallback(const base_pkg::Command::ConstPtr& msg) {
    // receive an odom command and add to the queue.
    cmdQueue.push(msg);
}

void lmMeasCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // receive a landmark detection and add to the queue.
    lmMeasQueue.push(msg);
}

void trueMapCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // set the true map to be used for localization mode.
    ukf.map = msg->data;  
    loadedTrueMap = true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ukf_node");
    ros::NodeHandle node("~");

    // read command line args from launch file.

    if (argc < 2) {
        std::cout << "UKF mode argument not provided. Using 'loc'.\n" << std::flush;
        ukf.ukfSlamMode = false;
    } else {
        std::string mode = argv[1];
        ukf.ukfSlamMode = (mode.compare("slam") == 0);
    }
    // read config parameters.
    float DT = readParams();
    // get the initial veh pose and init the ukf.
    ros::Subscriber initSub = node.subscribe("/truth/init_veh_pose", 1, initCallback);

    // subscribe to inputs.
    ros::Subscriber cmdSub = node.subscribe("/command", 100, cmdCallback);
    ros::Subscriber lmMeasSub = node.subscribe("/landmark", 100, lmMeasCallback);
    ros::Subscriber trueMapSub = node.subscribe("/truth/landmarks", 1, trueMapCallback);
    // publish state.
    statePub = node.advertise<base_pkg::UKFState>("/state/ukf", 1);

    // timer to update UKF at set frequency.
    ros::Timer ekfIterationTimer = node.createTimer(ros::Duration(DT), &ukfIterate, false);

    ros::spin();
    return 0;
}