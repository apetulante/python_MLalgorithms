#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:01:40 2018

@author: petulaa
"""

'''
Functions to do a binary-classifying neural network with any given number of hidden layers.
Some common activation functions can be found in the ML_helper.py file,
or custom ones can be passed but should follow the same format.
Loss is cross-entropy.
'''

import numpy as np

# general forward propagation, which takes as input the desired activation function
def forward(W,X,b,activ_func):
    Z = np.dot(W,X)+b
    A = activ_func(Z, deriv=0)
    return A, Z


# backpropagation for the hidden layers
def back(m,nf,n_nodes,A,W,Z,dA,activ_func):
    if n_nodes==1: # if this is the output layer, then dA was already fed in as dZ
        dZ = dA
    else: #otherwise, calculate it
        dZ = dA * activ_func(Z, deriv=1)
    dW = 1/m * np.dot(dZ,A.T)
    db = 1/m * np.sum(dZ,axis=1).reshape(n_nodes,1)
    dA_next = np.dot(W.T,dZ)
    return dZ, dW, db, dA_next

  
# calculation of the current cost
def calc_cost(A,y,m,W,lam=0):
    return (-1/m * np.sum((y*np.log(A[0]))+((1-y)*np.log(1-A[0])))) + ((lam/(2*m))*np.sum(np.square(W)))


'''
Function that loops through iterations and trains the NN
parameters are:
    X - input features, must have shape (# features, # examples)
    y - output data, must have shape (1, # examples)
    alpha - the learning rate
    iterations - the number of iterations to go over
    layers - the number/sizes of hidden layers
    output_func - function to use for the output layer
    hidden_func - function to use for hidden layers
    reg - whether or not to use regularization, "none" or "l2", default none
    lam - the regularization hyperparameter
    dropout - whether or not to use dropout, True or False
    prob - probability to use for keeping neurons for dropout
    verbose - whether or not to print updates as it trains. Default is no updates.
        entered as a integer, with the integer being the number of iterations between
        outputs (i.e verbose=1 prints the cost at every step)
'''
def calc_NN(X,y,alpha,iterations,layers,output_func,hidden_func,
            reg='none',lam=0, dropout=False, prob=0, verbose=0):
    # get shapes and parameters of the network from input params
    n_layers = len(layers)
    nf,m = X.shape
    
    # initialize the weights and biases of the output layer, using a uniform random
    # distrubtion
    W_out = np.random.uniform(low=-.2,high=.2,size=(1,layers[-1]))
    b_out = 0
    
    # initialize the weights and biases of the input layer
    W_hid = [np.random.uniform(low=-.2,high=.2,size=(layers[0],nf))]
    b_hid = [np.zeros((layers[0],1))]
    for i in range(1,len(layers)):
        W_hid.append(np.random.uniform(low=-.2,high=.2,size = (layers[i],layers[i-1])))
        b_hid.append(np.zeros((layers[i],1)))
        
    # Forward and backprop over all iterations
    cost = []
    for i in range(iterations): #loop over number of iterations
        if verbose != 0 and i%verbose == 1:
            print("Working on iteration %d, cost = %f" %(i,cost[-1]))    
    
        masks = []
        A, Z = forward(W_hid[0],X,b_hid[0],activ_func=hidden_func) # from input to first layer
        # check if we want to do dropout
        if dropout == True:
            D = np.random.uniform(low=0,high=1,size=A.shape)
            D[D>prob] = 0 # collapse the mask to zeroes/ones
            D[D<=prob] = 1
            masks.append(D) # store masks so we can use them in backprop later
            A = (A*D)/prob
        A_hid = [A] #store in arrays to use later
        Z_hid = [Z]
        for j in range(1,n_layers):
            A,Z = forward(W_hid[j],A,b_hid[j],activ_func=hidden_func) #between layers
            # check if we want to do dropout
            if dropout == True:
                D = np.random.uniform(low=0,high=1,size=A.shape)
                D[D>prob] = 0 # collapse the mask to zeroes/ones
                D[D<=prob] = 1
                masks.append(D) # store masks so we can use them in backprop later
                A = (A*D)/prob
            A_hid.append(A) #store in arrays to use later
            Z_hid.append(Z)
        A_out, Z_out = forward(W_out,A_hid[-1],b_out, activ_func=output_func) # from last layer to output
        
        # keep the cost at every iteration
        cost.append(calc_cost(A_out,y,m,W_out,lam))
        
        # do the backpropogation, over all layers
        dL_dZ = np.subtract(A_out,y)
        dZ_out, dW_out, db_out, dA = back(m,nf,1,A_hid[-1],W_out,Z_out,dL_dZ,output_func)
        dZ_hid, dW_hid, db_hid = [],[],[]
        for j in range(1,n_layers): #going backward from last to first later
            j=-j
            #between hidden layers
            if dropout == True: # if we are doing dropout
                dA = (masks[j]*dA)/prob
            dZ, dW, db, dA = back(m,nf,layers[j],A_hid[j-1],W_hid[j],Z_hid[j],dA, hidden_func)
            dZ_hid.append(dZ) #store in arrays to use later
            dW_hid.append(dW)
            db_hid.append(db)
        dZ, dW, db, dA = back(m,nf,layers[0],X,W_hid[0],Z_hid[0],dA, hidden_func) # from input to first layer
        dZ_hid.append(dZ) #store in arrays to use later
        dW_hid.append(dW)
        db_hid.append(db)
        
        # update the weights and biases
        for j in range(n_layers):
            # if no regularization, lam is just 0 and regularization term goes away
            W_hid[j] = W_hid[j] - alpha*(dW_hid[-j-1] + (lam/m)*W_hid[j]) # for all hidden layers
            b_hid[j] = b_hid[j] - alpha*db_hid[-j-1] # (-j-1) because gradient/weight arrays match each other in opposite order
        W_out = W_out - alpha*(dW_out + (lam/m)*W_out) # for output layer
        b_out = b_out - alpha*db_out
    
    return W_hid, b_hid, W_out, b_out, cost


# function to make predictions on test data, based on outputs of the NN
def predict(X,W_hid,W_out,b_hid,b_out,hid_func,out_func):
    output_func = eval(out_func) # need to know the functions used
    hidden_func = eval(hid_func)
    n_layers = len(W_hid)
    
    # forward propagate over all layers
    A, Z = forward(W_hid[0],X,b_hid[0],activ_func=hidden_func)
    for j in range(1,n_layers):
        A,Z = forward(W_hid[j],A,b_hid[j],activ_func=hidden_func)
    y_pred, Z_out = forward(W_out,A,b_out, activ_func=output_func)
        
    # collapse sigmoud output to binary yes/no    
    y_pred[0][np.where(y_pred[0] >= .5)[0]] = 1
    y_pred[0][np.where(y_pred[0] < .5)[0]] = 0
    
    return y_pred[0]