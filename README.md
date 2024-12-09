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

- **Maxime Sabbah (LAAS-CNRS):** Main developer and maintainer of the project, real-time pipeline, inverse kinematics, and general implementation of the library.
- **Kahina Chalabi (LAAS-CNRS):** Main developer and maintainer of the project, MMpose, and LSTM.
- **Mohamed Adjel (LAAS-CNRS):** Main developer, human modeling, and camera feeds handling.
- **Thomas Bousquet (LAAS-CNRS):** Features developer.
- **Vincent Bonnet (LAAS-CNRS / IPAL):** Project instructor.


## Citing RT-COSMIK


## License
BSD 2-Clause License

Copyright (c) 2024, LAAS-CNRS
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Project status
This repo is still under development, I am available at msabbah@laas.fr
