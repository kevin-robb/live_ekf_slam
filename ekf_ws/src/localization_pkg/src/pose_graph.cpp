#include "localization_pkg/filter.h"

// init the PoseGraph.
PoseGraph::PoseGraph() {
    // set filter type.
    this->type = FilterChoice::POSE_GRAPH_SLAM;
    ///\todo: initialization and whatnot.
}

void PoseGraph::readParams(YAML::Node config) {
    // setup all commonly-used params.
    Filter::readParams(config);
    // setup all filter-specific params, if any.
}

void PoseGraph::init(float x_0, float y_0, float yaw_0) {
    // set starting vehicle pose.
    ///\todo: change to using SE(2) representation for everything.
    this->x_t << x_0, y_0, yaw_0;
    // set initialized flag.
    this->isInit = true;
}

// Update the graph with the new information for this iteration.
void PoseGraph::update(base_pkg::Command::ConstPtr cmdMsg, std_msgs::Float32MultiArray::ConstPtr lmMeasMsg) {
    // update timestep.
    this->timestep += 1;
    ///\todo: do the thing.
}

///\todo: function(s) to do the optimization and solve the graph.