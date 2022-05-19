#include "ros/ros.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <queue>
#include <iostream>

#include "ukf_pkg/ukf.h"

// define queues for messages.
std::queue<geometry_msgs::Vector3::ConstPtr> odomQueue;
std::queue<std_msgs::Float32MultiArray::ConstPtr> lmMeasQueue;
// define state publisher.
ros::Publisher statePub;
// define UKF object from ukf.cpp class.
UKF ukf;

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
    float param = -2;
    ros::init(argc, argv, "ukf_node");
    ros::NodeHandle node("~");
    node.getParam("/DT", param);
    float DT = param; //std::stof(param);
    if (DT <= 0) {
        std::cout << "Using DT=0.05." << std::endl << std::flush;
        DT = 0.05;
    }

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