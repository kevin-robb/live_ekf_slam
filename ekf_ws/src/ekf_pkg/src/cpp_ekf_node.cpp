#include "ros/ros.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <queue>

#include "ekf_pkg/ekf.h"

// define queues for messages.
std::queue<geometry_msgs::Vector3::ConstPtr> odomQueue;
std::queue<std_msgs::Float32MultiArray::ConstPtr> lmMeasQueue;
// define EKF state publisher.
ros::Publisher statePub;
// define EKF object from ekf.cpp class.
EKF ekf;

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
    ekf.update(odomMsg, lmMeasMsg);
    // get the current state estimate.
    ekf_pkg::EKFState stateMsg = ekf.getState();
    // publish it.
    statePub.publish(stateMsg);
}

void odomCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    // receive an odom command and add to the queue.
    odomQueue.push(msg);
    // ROS_INFO("Received odom: [%s, %s]", msg->x.c_str(), msg->y.c_str());
}

void lmMeasCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // receive a landmark detection and add to the queue.
    lmMeasQueue.push(msg);
    // ROS_INFO("Received lm: [%s]", msg->data.c_str());
}

int main(int argc, char **argv) {
    std::string param;
    ros::init(argc, argv, "ekf_node");
    ros::NodeHandle node;
    node.getParam("DT", param);
    float DT = std::stof(param);

    // subscribe to EKF inputs.
    ros::Subscriber odomSub = node.subscribe("/odom", 100, odomCallback);
    ros::Subscriber lmMeasSub = node.subscribe("/landmark", 100, lmMeasCallback);
    // publish EKF state.
    ros::Publisher statePub = node.advertise<std_msgs::Float32MultiArray>("/ekf/state", 1);

    // timer to update EKF at set frequency.
    float DT = 0.05;
    ros::Timer ekfIterationTimer = node.createTimer(ros::Duration(DT), &ekfIterate, false);

    ros::spin();
    return 0;
}