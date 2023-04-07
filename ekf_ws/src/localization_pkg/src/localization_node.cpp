#include "ros/ros.h"
#include "geometry_msgs/Vector3.h"
#include "std_msgs/Float32MultiArray.h"
#include <queue>
#include <iostream>
#include <string>
#include <fstream>
#include <ros/package.h>
#include <boost/algorithm/string.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>

#include "localization_pkg/filter.h"

// define queues for messages.
std::queue<base_pkg::Command::ConstPtr> cmdQueue;
std::queue<std_msgs::Float32MultiArray::ConstPtr> lmMeasQueue;
// define state publisher.
ros::Publisher statePub;

// flag to wait for map to be received (for localization-only filters).
bool loadedTrueMap = false;

std::unique_ptr<Filter> filter = nullptr;

float readParams() {
    // read config parameters from file.
    std::string pkg_path = ros::package::getPath("base_pkg");
    YAML::Node config = YAML::LoadFile(pkg_path+"/config/params.yaml");

    const float DT = config["DT"].as<float>();

    // Setup filter as the chosen derived class type.
    const std::string filter_choice_str = config["FILTER"].as<std::string>();
    if (filter_choice_str == "ekf_slam") {
        filter = std::make_unique<EKF>();
    } else if (filter_choice_str == "ukf_slam") {
        filter = std::make_unique<UKF>();
    } else if (filter_choice_str == "ukf_loc") {
        filter = std::make_unique<UKF>();
        filter->type = FilterChoice::UKF_LOC; // Override default of UKF_SLAM.
    } else if (filter_choice_str == "pose_graph") {
        filter = std::make_unique<PoseGraph>();
    } else {
        std::cout << "Invalid filter choice in params.yaml." << std::endl << std::flush;
        exit(2);
    }

    // Setup params for the specified filter.
    filter->readParams(config);

    return DT;
}

void initCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    // receive the vehicle's initial position.
    float x_0 = msg->x;
    float y_0 = msg->y;
    float yaw_0 = msg->z;
    // init the filter.
    filter->init(x_0, y_0, yaw_0);
}

void iterate(const ros::TimerEvent& event) {
    if (!filter->isInit || cmdQueue.empty() || lmMeasQueue.empty()) {
        // wait for filter to init and to get command and measurement.
        return;
    }
    if (filter->type == FilterChoice::UKF_LOC && !loadedTrueMap) {
        // for localization-only filters, wait to get the map.
        return;
    }
    // get the next timestep's messages from the queues.
    base_pkg::Command::ConstPtr cmdMsg = cmdQueue.front();
    cmdQueue.pop();
    std_msgs::Float32MultiArray::ConstPtr lmMeasMsg = lmMeasQueue.front();
    lmMeasQueue.pop();
    // call the filter's update function.
    filter->update(cmdMsg, lmMeasMsg);

    // Not all filters will necessarily have the same (or any) state output.
    switch (filter->type) {
        case FilterChoice::EKF_SLAM: {
            base_pkg::EKFState stateMsg = filter->getEKFState();
            statePub.publish(stateMsg);
            break;
        }
        case FilterChoice::UKF_LOC: 
        case FilterChoice::UKF_SLAM: {
            base_pkg::UKFState stateMsg = filter->getUKFState();
            statePub.publish(stateMsg);
            break;
        }
        default: {
            ///\todo: need to publish something for state so the sim will keep going?
            break;
        }
    }
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
    filter->map = msg->data;  
    loadedTrueMap = true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "localization_node");
    ros::NodeHandle node("~");

    // read config parameters.
    float DT = readParams();
    // get the initial veh pose and init the filter.
    ros::Subscriber initSub = node.subscribe("/truth/init_veh_pose", 1, initCallback);

    // subscribe to filter inputs.
    ros::Subscriber cmdSub = node.subscribe("/command", 100, cmdCallback);
    ros::Subscriber lmMeasSub = node.subscribe("/landmark", 100, lmMeasCallback);

    // publish localization state.
    // Not all filters will necessarily have the same (or any) state output.
    switch (filter->type) {
        case FilterChoice::EKF_SLAM: {
            statePub = node.advertise<base_pkg::EKFState>("/state/ekf", 1);
            break;
        }
        case FilterChoice::UKF_LOC: 
        case FilterChoice::UKF_SLAM: {
            statePub = node.advertise<base_pkg::UKFState>("/state/ukf", 1);
            break;
        }
        default: {
            ///\todo: need to publish something for state so the sim will keep going?
            break;
        }
    }

    // timer to update filter at set frequency.
    ros::Timer iterationTimer = node.createTimer(ros::Duration(DT), &iterate, false);

    ros::spin();
    return 0;
}