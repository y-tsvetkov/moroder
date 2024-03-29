# Moroder - Competition Repo

This GitHub repo hosts all code used in the paper "Training quadruped robots to walk via evolution strategies and sinusoidal activation functions".
## Abstract
This project in the field of robotics demonstrates the application of a neural network that uses the sinusoidal activation function for the task of perturbed walking. The usage of such networks is analysed. Furthermore, the most common local minima and the methods of resolving them are presented. The small number of neurons allows the network to be deployed on a single microcontroller, narrowing the sim-to-real gap. The physical robot uses various design optimisations to perform highly dynamic gaits. The quadruped is capable of reaching speeds higher than 1 body length per second without an auxiliary power supply.
# Repository Overview
This repository contains everything needed to simulate and train a quadruped robot in PyBullet using TensorFlow. It includes the following packages:
```
.
|
+-- MoroderArduino/        Arduino code to run on real robot
|
+-- moroder_env/               PyBullet simulation environment
|
+-- optimise_cma.py         main training and optimisation script
|
+-- optimise_functions.py           functions for main script
|
.
```
# Hardware
The Fusion 360 design can be found at the following Google Drive link:

## https://drive.google.com/file/d/17Si-72igXzpSMtHh0uWlVNcXRKu1gA5d/view?usp=sharing
