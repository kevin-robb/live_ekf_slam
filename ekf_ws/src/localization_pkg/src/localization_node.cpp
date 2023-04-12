#include "ros/ros.h"
#include <ros/console.h>
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

// flag to wait for map to be received (for localization-only filters).
bool loadedTrueMap = false;

// filter(s) to run.
std::unique_ptr<Filter> filter;
std::unique_ptr<Filter> filter_secondary; // Second filter to run, if filter->filter_to_compare != NOT_SET.


float readParams() {
    std::string pkg_path = ros::package::getPath("base_pkg");
    YAML::Node config = YAML::LoadFile(pkg_path+"/config/params.yaml");
    // Get desired timer period.
    float DT = config["dt"].as<float>();

    // Setup filter as the chosen derived class type.
    std::string filter_choice_str = config["filter"].as<std::string>();
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
        throw std::runtime_error("Invalid filter choice in params.yaml.");
    }
    // Setup params for the specified filter.
    filter->readParams(config);

    // Setup secondary filter if applicable.
    if (filter->filter_to_compare == filter->type) {
        throw std::runtime_error("Cannot instantiate two instances of the same filter.");
    }
    switch (filter->filter_to_compare) {
        // Define allowed secondary filters.
        case FilterChoice::EKF_SLAM: {
            filter_secondary = std::make_unique<EKF>();
            break;
        }
        case FilterChoice::UKF_SLAM: {
            filter_secondary = std::make_unique<UKF>();
            break;
        }
        case FilterChoice::UKF_LOC: {
            filter_secondary = std::make_unique<UKF>();
            filter_secondary->type = FilterChoice::UKF_LOC; // Override default of UKF_SLAM.
            break;
        }
        case FilterChoice::NAIVE_COMMAND_PROPAGATION: {
            filter_secondary = std::make_unique<NaiveFilter>();
            break;
        }
        default: {
            // Do not instantiate secondary filter. Its type will remain NOT_SET.
            if (filter->type == FilterChoice::POSE_GRAPH_SLAM) {
                throw std::runtime_error("PGS requires a secondary filter.");
            }
            break;
        }
    }
    // Setup params for the secondary filter, if applicable.
    if (filter->filter_to_compare != FilterChoice::NOT_SET) {
        filter_secondary->readParams(config);
    }

    return DT;
}

void initCallback(const geometry_msgs::Vector3::ConstPtr& msg) {
    // receive the vehicle's initial position.
    float x_0 = msg->x;
    float y_0 = msg->y;
    float yaw_0 = msg->z;
    // wait for the params to be read and filter to be chosen.
    while (filter->type == FilterChoice::NOT_SET) {
        sleep(1);
    }
    // init the filter.
    filter->init(x_0, y_0, yaw_0);

    // if a secondary filter is defined, init it too.
    if (filter->filter_to_compare != FilterChoice::NOT_SET) {
        filter_secondary->init(x_0, y_0, yaw_0);
    }
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

    // first update the secondary filter (if applicable)
    if (filter->filter_to_compare != FilterChoice::NOT_SET) {
        filter_secondary->update(cmdMsg, lmMeasMsg);
        // inform the primary filter of this result.
        Eigen::Vector3d cur_veh_pose_est = filter_secondary->getStateVector();
        filter->updateNaiveVehPoseEstimate(cur_veh_pose_est(0), cur_veh_pose_est(1), cur_veh_pose_est(2));
    }

    // call the filter's update function.
    filter->update(cmdMsg, lmMeasMsg);

    ///\note: Not all filters will necessarily have the same (or any) state output.
    // Priotitize publishing secondary filter's state, since the main filter doesn't produce an online estimate.
    if (filter->filter_to_compare != FilterChoice::NOT_SET) {
        filter_secondary->publishState();
    } else {
        filter->publishState();
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

    // Init subscribers as soon as possible to avoid missing data.
    // subscribe to filter inputs.
    ros::Subscriber cmdSub = node.subscribe("/command", 100, cmdCallback);
    ros::Subscriber lmMeasSub = node.subscribe("/landmark", 100, lmMeasCallback);
    // get the initial veh pose and init the filter.
    ros::Subscriber initSub = node.subscribe("/truth/init_veh_pose", 1, initCallback);
    // get the true landmark map (used for localization-only filters).
    ros::Subscriber trueMapSub = node.subscribe("/truth/landmarks", 1, trueMapCallback);

    // read config parameters and setup the specific filter instance.
    float DT = readParams();

    // publish localization state.
    filter->setupStatePublisher(node);
    // if there is a secondary node, set up its publisher too.
    if (filter->filter_to_compare != FilterChoice::NOT_SET) {
        ROS_WARN_STREAM("LOC: Setting up state publisher for secondary filter.");
        filter_secondary->setupStatePublisher(node);
    }

    // timer to update filter at set frequency.
    ros::Timer iterationTimer = node.createTimer(ros::Duration(DT), &iterate, false);

    ros::spin();
    return 0;
}