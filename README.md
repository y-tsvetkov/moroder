# Moroder - Competition Repo
This GitHub repo hosts all code used in the paper "Training quadruped robots to walk via evolution strategies and sinusoidal activation functions".
## Abstract
This project in the field of robotics demonstrates the application of a neural network that uses the sinusoidal activation function for the task of perturbed walking. The usage of such networks is analysed. Furthermore, the most common local minima and the methods of resolving them are presented. The small number of neurons allows the network to be deployed on a single microcontroller, narrowing the sim-to-real gap. The physical robot uses various design optimisations to perform highly dynamic gaits. The quadruped is capable of reaching speeds higher than 1 body length per second without an auxiliary power supply.
# Repository Overview
This repository contains everything needed to simulate and control Maah in Gazebo 11 using ROS Noetic. It includes the following packages:
```
.
|
+-- moroder_hardware/              Fusion 360 design of robot
|
+-- MoroderArduino/        Arduino code to run on real robot
|
+-- moroder/               PyBullet simulation environment
|
+-- moroder_cma.py         main training and optimisation script
|
+-- functions.py           functions for moroder_cma.py
|
.
```
