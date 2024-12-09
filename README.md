# RT-COSMIK
***Real Time - Constrained and Open Source Multibodied Inverse Kinematics

## Getting started

This repo usage is with the following provided docker : https://gitlab.laas.fr/msabbah/gepetto-dev-container/-/tree/mmpose?ref_type=heads
Follow the installation procedure there

## Clone the repo in the docker 
```
cd workspace/ros_ws_src
git clone https://gitlab.laas.fr/msabbah/rt-cosmik.git
```

## Test and Deploy
All the usable code can be found in apps folder for running the pipelines and on cams_calibration for cameras calibration

For ros visualisation in rviz do the following command : 

```
roslaunch rt-cosmik start_viz.launch
```

## Authors and acknowledgment
Maxime Sabbah (LAAS-CNRS) : Main developper and maintainer of the project, Realtime pipeline, inverse kinematics and general implementation of the library
Kahina Chalabi (LAAS-CNRS) : Main developper and maintainer of the project, MMpose and LSTM 
Mohamed Adjel (LAAS-CNRS) : Main developper, Human modelling and camera feeds handling
Thomas Bousquet (LAAS-CNRS) : Features developper 
Vincent Bonnet (LAAS-CNRS / IPAL) : Project Instructor

## Citing RT-COSMIK


## License
For open source projects, say how it is licensed.

## Project status
This repo is still under development, I am available at msabbah@laas.fr
