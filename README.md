# logistic_donut
Solving the "donut" classification problem by hand using a simple logistic regression and a coordinates trick.

This is a toy problem in which I code a linear regression from scratch; I do not use any Machine Learning library.

First I build the problem: two sets of points are located in concentric rings around the 2D plane.
A logistic regression seperates data with a straight 1D line in the 2D plane.
Since this is not possible, we will use a trick: we transform the data into a 3D cone, using the distance from the center as the z coordinate.
Following this transformation, we can do gradient descent on a sigmoid function to get the seperation.

Visually, the weights we obtained define a 2D plane in the 3D space.

Inputs:
- None

Outputs:
- Returns None. 
- Plots the initial classficication problem (the 2D donut)
- Plots the crossentropy loss function evolution to make sure we converged to a solution
- Plots the transformed problem in the 3D space, where we see our solution plane seperate the data


The final classification rate is above 99% accurate.

