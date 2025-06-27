#!/bin/bash
source /opt/ros/noetic/setup.bash
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
roslaunch rosbridge_websocket_qtpc.launch
