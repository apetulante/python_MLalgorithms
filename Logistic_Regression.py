#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:42:26 2018

@author: petulaa
"""

import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def forward(X,W,b):
    Z = np.dot(W.T,X) + b
    A = sigmoid(Z)
    return Z, A
    
def backward(X,y,A,Z,m):
    dZ = np.subtract(A,y)
    dW = 1/m * np.dot(X,dZ.T)
    db = 1/m * np.sum(Z)
    return dW, db

def calc_cost(A,y,m):
    return -1/m * np.sum((y[0]*np.log(A[0]))+((1-y[0])*np.log(1-A[0]))) 

"""
The primary function. Used to fit the logistic regression. 
Takes as input parameters:
    X - the matrix of inputs, with shape (# features, # training samples)
    y - the array of outputs, with shape (1, # features)
    alpha - the learning rate
    iterations - the number of iterations to use
And it returns:
    W - the matrix of weights
    b - the bias
    cost - an array of the cost at each iteration
    verbose - whether or not to print updates as it trains. Default is no updates.
        entered as a integer, with the integer being the number of iterations between
        outputs (i.e verbose=1 prints the cost at every step)
"""
def calc_logReg(X,y,alpha,iterations,verbose=0):  
    # initialize weights and bias
    n,m = X.shape
    W = np.zeros(n).reshape(n,1)
    b = 1
    cost = []
    for i in range(iterations): #loop over number of iterations
        if verbose != 0 and i%verbose == 1: # optional printing during training
            print("Working on iteration %d, cost = %f" %(i,cost[-1]))
        # forward propogation
        Z, A = forward(X,W,b)
        cost.append(calc_cost(A,y,m))
        
        # backpropagation
        dW, db = backward(X,y,A,Z,m)
        
        # update weights
        W = W - alpha*dW
        b = b - alpha*db
    
    return W,b,cost

""" Function used to make predictions after the regression has been fit.
Takes as input:
    X - the matrix of examples to predict outputs for, with shape (# features, # training samples)
    W - the weight matrix from the fit
    b - the bias from the fit
"""
def predict(X,W,b):
    y_lin, y = forward(X,W,b)
    y[0][np.where(y[0] >= .5)[0]] = 1
    y[0][np.where(y[0] < .5)[0]] = 0
    return y[0]