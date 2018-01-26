#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:18:10 2018

@author: petulaa
"""

'''
A simple set of functions to do a linear regression using gradient descent.
Generalized to single or multiple.
Assumes features are already scaled and that a train-test split was already 
done, so the calc_fit function is given scaled training data only.
Functions to scale/train-test split can be found in ML_helper.py
'''

import numpy as np

# function to load in data and initialize necessary variables
def initialize(X_full,y_full):
    X_features = X_full #features within the data
    X_intercept = np.ones(len(X_features)).reshape(len(X_features),1) #the intercept parameter

    X = np.hstack((X_intercept,X_features)) #add intercept to feature matrix
    y = y_full.reshape(len(X_features),1) #the outcome data
    theta = np.ones((X.shape[1],1)) #starting weights
    
    m = X.shape[0] #number of features
    n = X.shape[1] - 1 #number of observations
    
    return X, y, theta, m, n

# function to calculate the cost
def calc_cost(X,y,theta,m):
    J = 1/(2*m) * np.dot((np.dot(X,theta) - y).T,(np.dot(X,theta) - y))
    return np.float(J)

# function to calculate the gradient
def calc_gradient(X,y,theta,m):
    del_J = (1/m) * (np.dot(X.T,np.dot(X,theta)) - np.dot(X.T,y))
    return del_J

# generalized function to do the regression
def calc_fit(X_full,y_full,alpha,iterations):
    X,y,theta, m, n = initialize(X_full,y_full)
    cost = [calc_cost(X,y,theta,m)]
    for i in range(iterations): #loop over number of iterations
        grad_J = calc_gradient(X,y,theta,m)
        for j in range(n+1): #update all weights
            theta[j] = theta[j] - alpha*grad_J[j]
        cost.append(calc_cost(X,y,theta,m)) #keep track of cost
        
    return theta, cost

def predict(fit, X):
    X_intercept = np.ones(len(X)).reshape(len(X),1) #the intercept parameter
    X = np.hstack((X_intercept,X)) #add intercept to feature matrix
    y = np.dot(fit.T,X.T)
    return y