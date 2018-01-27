#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:51:10 2018

@author: petulaa
"""

import numpy as np

# Function to do standard scaling on the data. Subtracts my the mean
# and normalizes by variance.
def standard_scale(X,y):
    #feature scaling
    X_scaled = X.copy()
    for i in range(1,X_scaled.shape[1]):
        mean = np.mean(X[:,i])
        std = np.std(X[:,i])
        X_scaled[:,i] = (X[:,i] - mean)/std

    return X_scaled, y


# Split the data into training and test set.
def train_test_split(X, y, test_size):
    n_test = int(np.ceil(test_size*len(y)))
    subset = np.arange(len(y))
    np.random.shuffle(subset)
    train_subset = subset[0:n_test]
    test_subset = subset[n_test:]
    
    Xtrain = X[train_subset]
    ytrain = y[train_subset]
    Xtest = X[test_subset]
    ytest = y[test_subset]
    
    return Xtest, ytest, Xtrain, ytrain


# Encode categorical values with one-hot encoding.
# Categorical values are assumed to be already with values ranging
# from 0 to n with the different categories, without skipping numbers
def one_hot(X, cols):
    cols = np.array(cols)
    N, D = X.shape
    
    for i in range(len(cols)):
        col = cols[i]
        n_vals = len(np.unique(X[:,col]))
        new_cols = np.zeros((N,n_vals))
        new_cols[np.arange(N),X[:,col].astype(int)] = 1
        X = np.hstack([X[:,0:col],new_cols,X[:,col+1:]])
        cols = cols + n_vals - 1
    return X