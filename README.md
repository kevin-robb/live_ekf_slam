# Overview

This is a fully custom ROS structure to perform online EKF-SLAM. I've designed a crude simulator, which accepts the commanded forward and angular velocity for the next timestep, and moves the "real" vehicle according to a specified noise profile. It then generates noisy measurements of all landmarks within some specified vision range and FOV, which each come in the form of an ID with range and bearing relative to the vehicle. (Note that data association is not performed, as each landmark measurement includes a unique ID.) 

The "world" is comprised of both an occupancy map and a list of landmark positions. The landmark detections are used for the SLAM filters, and the occupancy map is used for path planning to avoid collisions. I've not actually implemented any collision detection in the simulator, but we can visually check how it's performing. A plotting node allows anything to be visualized, including the true occupancy/collision map, the true landmark locations, the true current vehicle pose, and the current estimates of the vehicle and landmark positions.

I've made this very modular, so the filter itself can be swapped out for any that I want to make. So far, I have implemented the following filters:
 - Extended Kalman Filter (EKF) in C++ (Best performance out of all)
 - EKF in Python
 - Unscented Kalman Filter (UKF) in C++ (Not working great yet)
 - MCL Particle Filter in Python (Not optimized, so it's unbearably slow)

The trajectory that the vehicle pursues can also be created in a variety of ways. I have created a few general approaches:
 - Treat the landmark map as a "travelling salesman problem" and autonomously generate a full trajectory that ensures the vehicle will pass by all landmarks at some point during the runtime. The entire trajectory is known at the start, and is published for the filter/plotter one at a time.
 - Let the user click on the map somewhere, and use A* on the occupancy map to find a path there. If one exists, navigate along its trajectory either directly or using Pure Pursuit. 
 - For the IGVC-like maps, we only allow the planner access to a local area of the map in front of the vehicle, as our IGVC robot would have from its camera. Choose an arbitrary free point ahead on this local map, plan a path to it with A*, and navigate there with Pure Pursuit. This allows the vehicle to autonomously navigate endlessly around these courses as long as it doesn't get itself stuck.

All parameters, from noise profiles, to vision and motion constraints, to the map being used can be modified in `data_pkg/config/params.txt`. Additional occupancy maps can be added to `data_pkg/config/maps`. The method for landmark generation is specified on the command line when running any of my launch files by appending the argument `map:=CHOICE`. There are several options of this, but the main ones to note are `random`, `grid`, and `igvc1`.


<p float="center">
  <img src=images/igvc1_demo.gif height="500" />
  <img src=images/ekf_rand_demo.png height="500" />
</p>

My full derivation of the math for the EKF is [included as a pdf](docs/EKF_SLAM_Derivation.pdf). It contains separate derivations for EKF Localization, EKF Mapping, and the combined EKF-SLAM which is implemented in this codebase. I've implemented Online SLAM, meaning we simultaneously estimate the current vehicle pose and the landmark positions, but we don't retroactively estimate all previous vehicle poses. That larger problem is called Full SLAM.

# Running it
To run any of these demos, first run the following commands after cloning this repo.

    cd ekf_ws
    catkin_make
    source devel/setup.bash

Then choose one of the demos to run from any of the packages' `launch` directories. The launch files in `data_pkg` get the simulation working, and will be automatically launched by any of the others. Some possibilities are:

    roslaunch planning_pkg pursue_goal.launch map:=igvc1
    roslaunch ekf_pkg cpp_ekf_demo.launch map:=random


The `DT` parameter in `data_pkg/launch/demo_base.launch` sets the time between iterations. If you set this lower than your machine can handle, the filter will miss measurements and begin to fail. All other parameters can be modified in `data_pkg/config/params.txt`. Important ones you will want to change are `OCC_MAP_IMAGE` and `USE_LOCAL_PLANNER`.

