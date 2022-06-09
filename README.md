# Overview

This is a fully custom ROS structure to perform online EKF-SLAM. I've designed a crude simulator, which accepts the commanded forward and angular velocity for the next timestep, and moves the "real" vehicle according to a specified noise profile. It then generates noisy measurements of all landmarks within some specified vision range and FOV, which each come in the form of an ID with range and bearing relative to the vehicle. (Note that data association is not performed, as each landmark measurement includes a unique ID.) 

The "world" is comprised of both an occupancy map and a list of landmark positions. The landmark detections are used for the SLAM filters, and the occupancy map is used for path planning to avoid collisions. I've not actually implemented any collision detection in the simulator, but we can visually check how it's performing. A plotting node allows anything to be visualized, including the true occupancy/collision map, the true landmark locations, the true current vehicle pose, and the current estimates of the vehicle and landmark positions.

I've made this very modular, so the filter itself can be swapped out for any that I want to make. So far, I have implemented the following filters:
 - Extended Kalman Filter (EKF) SLAM in both C++ and Python3.
 - Unscented Kalman Filter (UKF) Localization and SLAM in C++.
 - MCL Particle Filter in Python3 (not working yet lol).

The trajectory that the vehicle pursues can also be generated in a variety of ways. I have implemented a few approaches:
 - Treat the landmark map as a "travelling salesman problem" and autonomously generate a full trajectory that ensures the vehicle will pass by all landmarks at some point during the runtime. The entire trajectory is known at the start, and is published for the filter/plotter one at a time. This is the default mode for the filter demos, since no interaction from the user is required, and it will move around the full map.
 - Let the user click on the map somewhere, and use A* on the occupancy map to find a path there. If one exists, navigate along its trajectory either directly or using Pure Pursuit. 
 - Only allow the planner access to a local area of the map in front of the vehicle, as our IGVC robot would have from its camera. It chooses an arbitrary free point ahead on this local map, plans a path to it with A*, and navigates there with Pure Pursuit. This allows the vehicle to autonomously navigate endlessly around looped track courses as long as it doesn't get itself stuck.

EKF-SLAM           |       EKF-SLAM
:-------------------------:|:-------------------------:
![Vehicle following a precomputed TSP-trajectory and performing EKF-SLAM on a map with landmarks placed randomly.](images/ekf_rand_demo.gif)  |  ![Vehicle following a precomputed TSP-trajectory and performing EKF-SLAM on a map with landmarks placed in a grid.](images/ekf_grid_demo.gif)

Local Planning           |      UKF-SLAM
:-------------------------:|:-------------------------:
![IGVC-like map with the vehicle performing EKF-SLAM to localize on a known occupancy grid, and doing local path planning to progress through the course.](images/igvc1_demo.gif)  |  ![Vehicle following a precomputed TSP-trajectory and performing UKF-SLAM on a map with randomly-placed landmarks.](images/ukf_slam_demo.gif)

Pursuing Clicked Point           |       Pursuing Clicked Point
:-------------------------:|:-------------------------:
![Vehicle pursuing a point clicked on the map, while performing EKF-SLAM to track its position.](images/pursuing_clicked_pt.gif)  |  ![Vehicle finding a path to a clicked point with A* and navigating with Pure Pursuit.](images/pursuing_clicked_pt_bldg.gif)


Some key parameters are set using command line arguments, which have different default values in each launch file to ensure the proper setup for the different demos. These can also be changed when running the launch files by appending `arg=choice` (e.g. `landmark_map:=random`). All other parameters can be modified in `data_pkg/config/params.txt`. Additional occupancy maps can be added to `data_pkg/config/maps`.

My full derivation of the math for the EKF is [included as a pdf](docs/EKF_SLAM_Derivation.pdf). It contains separate derivations for EKF Localization, EKF Mapping, and the combined EKF-SLAM which is implemented in this codebase. I've implemented Online SLAM, meaning we simultaneously estimate the current vehicle pose and the landmark positions, but we don't retroactively estimate all previous vehicle poses. That larger problem is called Full SLAM.

# Running it
To run any of these demos, first run the following commands after cloning this repo.

    cd ekf_ws
    catkin_make
    source devel/setup.bash

Then choose one of the demos to run the `data_pkg/launch` directory. The `sim_base.launch` is run automatically by all others to get the simulation working. Some that give good results are:

    roslaunch base_pkg igvc1.launch
    roslaunch base_pkg ekf_demo.launch
    roslaunch base_pkg ekf_demo.launch landmark_map:=grid
    roslaunch base_pkg ukf_demo.launch

Note: The `DT` parameter in `data_pkg/config/params.txt` sets the time between iterations. Lowering this value makes everything run faster, but if you set this lower than your machine can handle, it will be very choppy and potentially fail.

