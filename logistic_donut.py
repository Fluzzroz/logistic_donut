# logisitc regression classifier for the donut problem.
#
# the notes for this class can be found at: 
# https://deeplearningcourses.com/c/data-science-logistic-regression-in-python
# https://www.udemy.com/data-science-logistic-regression-in-python

# from __future__ import print_function, division
# from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

# distance from origin is radius + random normal
# angle theta is uniformly distributed between (0, 2pi)
R1 = np.random.randn(N//2) + R_inner
theta1 = 2*np.pi*np.random.random(N//2)
X_inner = np.concatenate([[R1 * np.cos(theta1)], [R1 * np.sin(theta1)]]).T
#shape of R1 * np.cos(theta) is (500,) (equivalent to either 500x1 or 1x500;
#since the concatenate default is axis=0, the shape of X_inner before the .T
# is (2,500), so each column is a pair of polar coordinates. Then the transpose

R2 = np.random.randn(N//2) + R_outer
theta2 = 2*np.pi*np.random.random(N//2)
X_outer = np.concatenate([[R2 * np.cos(theta2)], [R2 * np.sin(theta2)]]).T
#again, concatenate default is axis=0, so Shape(500,2) + (500,2) = (1000,2)


X = np.concatenate([ X_inner, X_outer ])
T = np.array([0]*(N//2) + [1]*(N//2)) # labels: first 50 are 0, last 50 are 1

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()



# add a column of ones for the bias term
ones = np.ones((N, 1))

# add a column of r = sqrt(x^2 + y^2)
#   the product is element by element, NOT a matrix product
#   the sum adds column 1 to column 2 for each row, which are X_inner and X_outer

r = np.sqrt( (X * X).sum(axis=1) ).reshape(-1, 1)
Xb = np.concatenate((ones, r, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 2)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


# let's do gradient descent 100 times
learning_rate = 0.0001
error = []
for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 500 == 0:
        print(e)

    # gradient descent weight udpate with regularization
    w += learning_rate * ( Xb.T.dot(T - Y) - 0.1*w )

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy per iteration")
plt.show()

print("Final w:", w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)
#that's a complicated way of writing np.mean(T == np.round(Y))
#   roundY turns the almost 1s and almost 0s to 1s and 0s
#   abs( T - Y ) is like T!= Y ; the abs make all differences positives
#   sum / N is the mean
#   1 - != is the same as == since sum of prob is 1


#plotting the 3D plane obtained from the weights
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-15, 15, 0.1)
y = np.arange(-15, 15, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)


#using the equation of a plane, we can compute the seperation plane
# that our weights generated
zz = -1/w[1] * (w[2]*xx + w[3]*yy + w[0])

ax.scatter(Xb[:, 2], Xb[:, 3], Xb[:,1], c=T)

#... and visualize it
ax.plot_surface(xx, yy, zz)
plt.show()

