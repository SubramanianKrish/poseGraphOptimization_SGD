# 2D Pose Graph Optimization
This repository is an implementation of 2D (i.e, 3 degrees of freedom - x,y,theta) pose graph optimization in Python.  
A modified gradient descent as authored by Olson et. al. found in refs is implemented.  
The code has been tested on the Intel and Manhattan dataset's TORO version.  

## Odometry motion for Manhattan data
![Odom M3500](./output/m3500_odom.gif)

## Graph optimization for M3500
![Optim M3500](./output/M3500/m3500_200_iterations.gif)

## Ground truth M3500 data
![Manhattan dataset](./output/M3500gt.png)

## Graph optimization for Intel data
![Optim Intel](./output/intel/intel_200_iterations.gif)

## Ground truth Intel data
![Intel dataset](./output/intel_gt.png)
