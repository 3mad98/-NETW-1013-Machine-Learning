# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 10:36:36 2021

@author: emad_
"""

# used for manipulating directory paths
import os
import sys

# Scientific and vector computation for python
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold

import pandas as pd

# Plotting library
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces
# tells matplotlib to embed plots within the notebook
# %matplotlib inline

# Load data
#data = np.loadtxt(os.path.join('.\ex1data2.txt'), delimiter=',')
#data = np.loadtxt('house_prices_data_training_data.csv', delimiter=',')
#data=pd.read_table('E:\Desktop\10th\(NETW 1013) Machine Learning (2350)\Assignment Take 2\house_prices_data_training_data.cvs', delim_whitespace=True, header=None)
#data = pd.read_csv (r'house_prices_data_training_data.csv')

data = genfromtxt('house_prices_data_training_data.csv', delimiter=',')

X = data[1:18000,3:]
y = data[1:18000, 2]
m=y.size

X2 = np.square(X)
X2=np.concatenate((X, X2), axis=1)

X3 = np.power(X, 3)
X3=np.concatenate((X2, X3), axis=1)

X4 = np.power(X, 4)
X4=np.concatenate((X3, X4), axis=1)

X5 = np.power(X, 5)
X5=np.concatenate((X4, X5), axis=1)

Xtrain=X[0:10800,:]
Xcv=X[10800:14400,:]
Xtest=X[14400:18000,:]

X2train=X2[0:10800,:]
X2cv=X2[10800:14400,:]
X2test=X2[14400:18000,:]

X3train=X3[0:10800,:]
X3cv=X3[10800:14400,:]
X3test=X3[14400:18000,:]

X4train=X4[0:10800,:]
X4cv=X4[10800:14400,:]
X4test=X4[14400:18000,:]

X5train=X5[0:10800,:]
X5cv=X5[10800:14400,:]
X5test=X5[14400:18000,:]

ytrain=y[0:10800,]
ycv=y[10800:14400,]
ytest=y[14400:18000,]


def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data 
    points and gives the figure axes labels of population and profit.
    
    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.
    
    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.    
    
    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You 
    can also set the marker edge color using the `mec` property.
    """

    fig = pyplot.figure()  # open a new figure
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    #pyplot.ylabel('Profit in $10,000')
    #pyplot.xlabel('Population of City in 10,000s')
    

def  featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    # =========================== YOUR CODE HERE =====================
    mu=[np.mean(X[:,0]),np.mean(X[:,1]),np.mean(X[:,2]),np.mean(X[:,3]),np.mean(X[:,4]),
        np.mean(X[:,5]),np.mean(X[:,6]),np.mean(X[:,7]),np.mean(X[:,8]),np.mean(X[:,9]),
        np.mean(X[:,10]),np.mean(X[:,11]),np.mean(X[:,12]),np.mean(X[:,13]),np.mean(X[:,14]),
        np.mean(X[:,15]),np.mean(X[:,16]),np.mean(X[:,17])]
    
    sigma=[np.std(X[:,0]),np.std(X[:,1]),np.std(X[:,2]),np.std(X[:,3]),np.std(X[:,4]),
        np.std(X[:,5]),np.std(X[:,6]),np.std(X[:,7]),np.std(X[:,8]),np.std(X[:,9]),
        np.std(X[:,10]),np.std(X[:,11]),np.std(X[:,12]),np.std(X[:,13]),np.std(X[:,14]),
        np.std(X[:,15]),np.std(X[:,16]),np.std(X[:,17])]
    X_norm=(X-mu)/sigma
    # ================================================================
    return X_norm, mu, sigma

def  featureNormalize2(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu. 
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation 
    in sigma. 
    
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature. 
    
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(0)
    sigma = np.zeros(0)

    # =========================== YOUR CODE HERE =====================
    for i in range(0,X.shape[1]):
        mu=np.concatenate([mu,[np.mean(X[:,i])]], axis=0)
        sigma=np.concatenate([sigma,[np.std(X[:,i])]], axis=0)
    
    X_norm=(X-mu)/sigma
    # ================================================================
    return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    # ======================= YOUR CODE HERE ===========================
    
    h=np.dot(X,theta)
    tmp=np.square(h-y)
    sum=np.sum(tmp)
    #i=0
    #for i in range(0,m-1):
     #   sum+=tmp[i]
    J=sum/(2*m)
    
    # ==================================================================
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
        hypothesis=0
        #theta=0
        #tmp1=0
        #tmp1=alpha*(np.dot(X,theta))-y
        #theta=theta-(tmp1.dot(X))/m
        theta=theta-alpha*(np.dot(X,theta)-y).dot(X)/m

        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))
    
    return theta, J_history
    
    
    





    """
    Computes the closed-form solution to linear regression using the normal equations.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        The value at each data point. A vector of shape (m, ).
    
    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).
    
    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.
    
    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    theta = np.zeros(X.shape[1])
    
    # ===================== YOUR CODE HERE ============================
    tmp1=np.linalg.pinv(np.dot(X.T,X))
    tmp2=np.dot(tmp1,X.T)
    theta=np.dot(tmp2,y)
    # =================================================================
    return theta

#plotData(X[:,0], y)
#plotData(X[:,1], y)
#plotData(X[:,2], y)
#plotData(X[:,3], y)
#plotData(X[:,4], y)
#plotData(X[:,5], y)
#plotData(X[:,6], y)
#plotData(X[:,7], y)
#plotData(X[:,8], y)
#plotData(X[:,9], y)
#plotData(X[:,10], y)
#plotData(X[:,11], y)
#plotData(X[:,12], y)
#plotData(X[:,13], y)
#plotData(X[:,14], y)
#plotData(X[:,15], y)
#plotData(X[:,16], y)
#plotData(X[:,17], y)







X_norm, mu, sigma =featureNormalize(X)
#print('Computed mean:', mu)
#print('Computed standard deviation:', sigma)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
alpha = 0.1
num_iters = 100
theta = np.zeros(19)
print(computeCostMulti(X, y, theta))
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
print(computeCostMulti(X, y, theta))

print("########")


#Xtrain_norm, mutrain, sigmatrain =featureNormalize(Xtrain)
#Xcv_norm, mucv, sigmacv =featureNormalize(Xcv)
#Xtest_norm, mutest, sigmatest =featureNormalize(Xtest)
#Xtrain = np.concatenate([np.ones((10800, 1)), Xtrain_norm], axis=1)
#Xcv = np.concatenate([np.ones((3600, 1)), Xcv_norm], axis=1)
#Xtest = np.concatenate([np.ones((3599, 1)), Xtest_norm], axis=1)

Xtrain=X[0:10800,:]
Xcv=X[10800:14400,:]
Xtest=X[14400:18000,:]

theta = np.zeros(19)
print(computeCostMulti(Xtrain, ytrain, theta))
theta, J_history = gradientDescentMulti(Xtrain, ytrain, theta, alpha, num_iters)
print(computeCostMulti(Xtrain, ytrain, theta),"train error 1")

print(computeCostMulti(Xcv, ycv, theta),"cv error 1")
print("#########")
############

X2train_norm, mu2, sigma2 =featureNormalize2(X2train)
X2cv_norm, mu2cv, sigma2cv =featureNormalize2(X2cv)
X2test_norm, mu2test, sigma2test =featureNormalize2(X2test)
X2train = np.concatenate([np.ones((10800, 1)), X2train_norm], axis=1)
X2cv = np.concatenate([np.ones((3600, 1)), X2cv_norm], axis=1)
X2test = np.concatenate([np.ones((3599, 1)), X2test_norm], axis=1)


theta2 = np.zeros(37)
print(computeCostMulti(X2train, ytrain, theta2))
theta2, J_history2 = gradientDescentMulti(X2train, ytrain, theta2, alpha, num_iters)
print(computeCostMulti(X2train, ytrain, theta2),"train error 2")

print(computeCostMulti(X2cv, ycv, theta2),"cv error 2")

print("#########")
############

X3train_norm, mu3, sigma3 =featureNormalize2(X3train)
X3cv_norm, mu3cv, sigma3cv =featureNormalize2(X3cv)
X3test_norm, mu3test, sigma3test =featureNormalize2(X3test)
X3train = np.concatenate([np.ones((10800, 1)), X3train_norm], axis=1)
X3cv = np.concatenate([np.ones((3600, 1)), X3cv_norm], axis=1)
X3test = np.concatenate([np.ones((3599, 1)), X3test_norm], axis=1)

theta3 = np.zeros(55)
print(computeCostMulti(X3train, ytrain, theta3))
theta3, J_history3 = gradientDescentMulti(X3train, ytrain, theta3, alpha, num_iters)
print(computeCostMulti(X3train, ytrain, theta3),"train error 3")

print(computeCostMulti(X3cv, ycv, theta3),"cv error 3")

print("#########")
############

X4train_norm, mu4, sigma4 =featureNormalize2(X4train)
X4cv_norm, mu4cv, sigma4cv =featureNormalize2(X4cv)
X4test_norm, mu4test, sigma4test =featureNormalize2(X4test)
X4train = np.concatenate([np.ones((10800, 1)), X4train_norm], axis=1)
X4cv = np.concatenate([np.ones((3600, 1)), X4cv_norm], axis=1)
X4test = np.concatenate([np.ones((3599, 1)), X4test_norm], axis=1)

theta4 = np.zeros(73)
print(computeCostMulti(X4train, ytrain, theta4))
theta4, J_history4 = gradientDescentMulti(X4train, ytrain, theta4, alpha, num_iters)
print(computeCostMulti(X4train, ytrain, theta4),"train error 4")

print(computeCostMulti(X4cv, ycv, theta4),"cv error 4")

print("#########")
############

X5train_norm, mu5, sigma5 =featureNormalize2(X5train)
X5cv_norm, mu5cv, sigma5cv =featureNormalize2(X5cv)
X5test_norm, mu5test, sigma5test =featureNormalize2(X5test)
X5train = np.concatenate([np.ones((10800, 1)), X5train_norm], axis=1)
X5cv = np.concatenate([np.ones((3600, 1)), X5cv_norm], axis=1)
X5test = np.concatenate([np.ones((3599, 1)), X5test_norm], axis=1)

theta5 = np.zeros(91)
print(computeCostMulti(X5train, ytrain, theta5))
theta5, J_history5 = gradientDescentMulti(X5train, ytrain, theta5, alpha, num_iters)
print(computeCostMulti(X5train, ytrain, theta5),"train error 5")

print(computeCostMulti(X5cv, ycv, theta5),"cv error 5")
print("#########")

print(computeCostMulti(Xtest, ytest, theta),"test error 1")
print(computeCostMulti(X2test, ytest, theta2),"test error 2")
print(computeCostMulti(X3test, ytest, theta3),"test error 3")
print(computeCostMulti(X4test, ytest, theta4),"test error 4")
print(computeCostMulti(X5test, ytest, theta5),"test error 5")

kf = KFold(n_splits=5)
kf.get_n_splits(X)
kf.split(X)
print("#########")
count1=0
for train_index, test_index in kf.split(X):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     theta, J_history = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters)
     print(computeCostMulti(X_test, y_test, theta),"test error 1st hyp Kfolding")
     count1+=computeCostMulti(X_test, y_test, theta)

kf = KFold(n_splits=5)
kf.get_n_splits(X2)
kf.split(X2)
count2=0
X2_norm, mu2, sigma2 =featureNormalize2(X2)
X2 = np.concatenate([np.ones((m, 1)), X2_norm], axis=1)
for train_index2, test_index2 in kf.split(X2):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train2, X_test2 = X2[train_index2], X2[test_index2]
     y_train2, y_test2 = y[train_index2], y[test_index2]
     theta2, J_history2 = gradientDescentMulti(X_train2, y_train2, theta2, alpha, num_iters)
     print(computeCostMulti(X_test2, y_test2, theta2),"test error 2nd hyp Kfolding")
     count2+=computeCostMulti(X_test2, y_test2, theta2)    

kf = KFold(n_splits=5)
kf.get_n_splits(X3)
kf.split(X3)
count3=0
X3_norm, mu3, sigma3 =featureNormalize2(X3)
X3 = np.concatenate([np.ones((m, 1)), X3_norm], axis=1)
for train_index3, test_index3 in kf.split(X3):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train3, X_test3 = X3[train_index3], X3[test_index3]
     y_train3, y_test3 = y[train_index3], y[test_index3]
     theta3, J_history3 = gradientDescentMulti(X_train3, y_train3, theta3, alpha, num_iters)
     print(computeCostMulti(X_test3, y_test3, theta3),"test error 3rd hyp Kfolding")
     count3+=computeCostMulti(X_test3, y_test3, theta3) 
     

kf = KFold(n_splits=5)
kf.get_n_splits(X4)
kf.split(X4)
count4=0
X4_norm, mu4, sigma4 =featureNormalize2(X4)
X4 = np.concatenate([np.ones((m, 1)), X4_norm], axis=1)
for train_index4, test_index4 in kf.split(X4):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train4, X_test4 = X4[train_index4], X4[test_index4]
     y_train4, y_test4 = y[train_index4], y[test_index4]
     theta4, J_history4 = gradientDescentMulti(X_train4, y_train4, theta4, alpha, num_iters)
     print(computeCostMulti(X_test4, y_test4, theta4),"test error 4th hyp Kfolding")
     count4+=computeCostMulti(X_test4, y_test4, theta4) 




alpha = 0.01
num_iters = 100
kf = KFold(n_splits=5)
kf.get_n_splits(X5)
kf.split(X5)
count5=0
X5_norm, mu5, sigma5 =featureNormalize2(X5)
X5 = np.concatenate([np.ones((m, 1)), X5_norm], axis=1)
for train_index5, test_index5 in kf.split(X5):
     #print("TRAIN:", train_index, "TEST:", test_index)
     X_train5, X_test5 = X5[train_index5], X5[test_index5]
     y_train5, y_test5 = y[train_index5], y[test_index5]
     theta5, J_history5 = gradientDescentMulti(X_train5, y_train5, theta5, alpha, num_iters)
     print(computeCostMulti(X_test5, y_test5, theta5),"test error 5th hyp Kfolding")
     count5+=computeCostMulti(X_test5, y_test5, theta5) 
     
print("##################")     
print(count1/5, "average error of 1st hyp Kfolding")
print(count2/5, "average error of 2nd hyp Kfolding")
print(count3/5, "average error of 3rd hyp Kfolding")
print(count4/5, "average error of 4th hyp Kfolding")
print(count5/5, "average error of 5th hyp Kfolding")

def computeCostMultiRegularized(X, y, theta, lambda_):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    Returns
    -------
    J : float
        The value of the cost function. 
    
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # You need to return the following variable correctly
    J = 0
    # ======================= YOUR CODE HERE ===========================
    
    h=np.dot(X,theta)
    tmp=np.square(h-y)
    sum=np.sum(tmp)
    sum2=np.sum(np.square(theta))
    #i=0
    #for i in range(0,m-1):
     #   sum+=tmp[i]
    J=sum/(2*m) +sum2*lambda_/(2*m)
    
    # ==================================================================
    return J


def gradientDescentMultiRegularized(X, y, theta, alpha, num_iters,lambda_):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
        
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    
    alpha : float
        The learning rate for gradient descent. 
    
    num_iters : int
        The number of iterations to run gradient descent. 
    
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    
    J_history : list
        A python list for the values of the cost function after each iteration.
    
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of 
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0] # number of training examples
    
    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()
    
    J_history = []
    for i in range(num_iters):
        # ======================= YOUR CODE HERE ==========================
        hypothesis=0
        #theta=0
        #tmp1=0
        #tmp1=alpha*(np.dot(X,theta))-y
        #theta=theta-(tmp1.dot(X))/m
        theta=theta-alpha*((np.dot(X,theta)-y).dot(X)+lambda_*theta)/m

        # =================================================================
        
        # save the cost J in every iteration
        J_history.append(computeCostMultiRegularized(X, y, theta,lambda_))
    
    return theta, J_history


print("##################")     
alpha = 0.1
num_iters = 1000
X = data[1:18000,3:]
y = data[1:18000, 2]

X2 = np.square(X)
X2=np.concatenate((X, X2), axis=1)

X3 = np.power(X, 3)
X3=np.concatenate((X2, X3), axis=1)

X4 = np.power(X, 4)
X4=np.concatenate((X3, X4), axis=1)

X5 = np.power(X, 5)
X5=np.concatenate((X4, X5), axis=1)

X2train=X2[0:10800,:]
X2cv=X2[10800:14400,:]
X2test=X2[14400:18000,:]

X3train=X3[0:10800,:]
X3cv=X3[10800:14400,:]
X3test=X3[14400:18000,:]

X4train=X4[0:10800,:]
X4cv=X4[10800:14400,:]
X4test=X4[14400:18000,:]

X5train=X5[0:10800,:]
X5cv=X5[10800:14400,:]
X5test=X5[14400:18000,:]

ytrain=y[0:10800,]
ycv=y[10800:14400,]
ytest=y[14400:18000,]
m=y.size
X_norm, mu, sigma =featureNormalize(X)
#print('Computed mean:', mu)
#print('Computed standard deviation:', sigma)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
Xtrain=X[0:10800,:]
Xcv=X[10800:14400,:]
Xtest=X[14400:18000,:]


theta = np.zeros(19)
lambda_=0.16
print(computeCostMultiRegularized(Xtrain, ytrain, theta,lambda_))
theta, J_history = gradientDescentMultiRegularized(Xtrain, ytrain, theta, alpha, num_iters,lambda_)
print(computeCostMultiRegularized(Xtrain, ytrain, theta,lambda_),"train error regularized 1")

print(computeCostMulti(Xcv, ycv, theta),"cv error regularized 1")
print("#########")
############


X2train_norm, mu2, sigma2 =featureNormalize2(X2train)
X2cv_norm, mu2cv, sigma2cv =featureNormalize2(X2cv)
X2test_norm, mu2test, sigma2test =featureNormalize2(X2test)
X2train = np.concatenate([np.ones((10800, 1)), X2train_norm], axis=1)
X2cv = np.concatenate([np.ones((3600, 1)), X2cv_norm], axis=1)
X2test = np.concatenate([np.ones((3599, 1)), X2test_norm], axis=1)


theta2 = np.zeros(37)
print(computeCostMultiRegularized(X, y, theta, lambda_))
theta2, J_history2 = gradientDescentMultiRegularized(X2train, ytrain, theta2, alpha, num_iters,lambda_)
print(computeCostMultiRegularized(X2train, ytrain, theta2,lambda_),"train error regularized 2")

print(computeCostMultiRegularized(X2cv, ycv, theta2,lambda_),"cv error regularized 2")

print("#########")
############

X3train_norm, mu3, sigma3 =featureNormalize2(X3train)
X3cv_norm, mu3cv, sigma3cv =featureNormalize2(X3cv)
X3test_norm, mu3test, sigma3test =featureNormalize2(X3test)
X3train = np.concatenate([np.ones((10800, 1)), X3train_norm], axis=1)
X3cv = np.concatenate([np.ones((3600, 1)), X3cv_norm], axis=1)
X3test = np.concatenate([np.ones((3599, 1)), X3test_norm], axis=1)

theta3 = np.zeros(55)
print(computeCostMultiRegularized(X3train, ytrain, theta3,lambda_))
theta3, J_history3 = gradientDescentMultiRegularized(X3train, ytrain, theta3, alpha, num_iters,lambda_)
print(computeCostMultiRegularized(X3train, ytrain, theta3,lambda_),"train error regularized 3")

print(computeCostMultiRegularized(X3cv, ycv, theta3,lambda_),"cv error regularized 3")

print("#########")
############

X4train_norm, mu4, sigma4 =featureNormalize2(X4train)
X4cv_norm, mu4cv, sigma4cv =featureNormalize2(X4cv)
X4test_norm, mu4test, sigma4test =featureNormalize2(X4test)
X4train = np.concatenate([np.ones((10800, 1)), X4train_norm], axis=1)
X4cv = np.concatenate([np.ones((3600, 1)), X4cv_norm], axis=1)
X4test = np.concatenate([np.ones((3599, 1)), X4test_norm], axis=1)

theta4 = np.zeros(73)
print(computeCostMultiRegularized(X4train, ytrain, theta4,lambda_))
theta4, J_history4 = gradientDescentMultiRegularized(X4train, ytrain, theta4, alpha, num_iters,lambda_)
print(computeCostMultiRegularized(X4train, ytrain, theta4,lambda_),"train error regularized 4")

print(computeCostMultiRegularized(X4cv, ycv, theta4,lambda_),"cv error regularized 4")

print("#########")
############

X5train_norm, mu5, sigma5 =featureNormalize2(X5train)
X5cv_norm, mu5cv, sigma5cv =featureNormalize2(X5cv)
X5test_norm, mu5test, sigma5test =featureNormalize2(X5test)
X5train = np.concatenate([np.ones((10800, 1)), X5train_norm], axis=1)
X5cv = np.concatenate([np.ones((3600, 1)), X5cv_norm], axis=1)
X5test = np.concatenate([np.ones((3599, 1)), X5test_norm], axis=1)

theta5 = np.zeros(91)
print(computeCostMultiRegularized(X5train, ytrain, theta5,lambda_))
theta5, J_history5 = gradientDescentMultiRegularized(X5train, ytrain, theta5, alpha, num_iters,lambda_)
print(computeCostMultiRegularized(X5train, ytrain, theta5,lambda_),"train error regularized 5")

print(computeCostMultiRegularized(X5cv, ycv, theta5,lambda_),"cv error regularized 5")
print("#########")

print(computeCostMulti(Xtest, ytest, theta),"test error regularized 1")
print(computeCostMulti(X2test, ytest, theta2),"test error regularized 2")
print(computeCostMulti(X3test, ytest, theta3),"test error regularized 3")
print(computeCostMulti(X4test, ytest, theta4),"test error regularized 4")
print(computeCostMulti(X5test, ytest, theta5),"test error regularized 5")