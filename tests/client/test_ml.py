#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:37:32 2022

@author: meracoda
"""


import math

import numpy as np
import operator
import tinychain as tc
import unittest

from functools import reduce
from testutils import ClientTest
import tinychain.ml.mlr import MultipleLinearRegression 

Dense = tc.tensor.Dense
HOST=tc.host.HOST('http:127.0.0.1:8702')
ENDPOINT="transact/hypothetical"

def sample_data():
    '''
    it creates sample data for tests.
    ===========
    Returns
    -------
    X_train : np.array
        Features to be trained on.
    X_test : np.array
        Features to be testsed on.
    y_train : TYPE
        Labels to be trained on.
    y_test : TYPE
        Labels to be tested on.

    '''
    np.random.seed(123)
    n_samples = 1000           # Number of data samples
    n_features = 2             # Number of features
    x = np.random.uniform(0.0, 1.0, (n_samples, n_features))
    X = np.hstack((np.ones((n_samples, 1)), x)) 
    mu, sigma = 0, 2           # Mean and standard deviation
    noise = np.random.normal(mu, sigma, (n_samples, 1))
    beta = np.array([30, -10, 70])
    beta = beta.reshape(len(beta), 1)
    y = X.dot(beta) + noise    # Actual y
    X_train,X_test,y_train,y_test=x[:-300],x[-300:],y[:-300],y[-300:]
    return X_train,X_test,y_train,y_test
    
    
class MLTests(ClientTest):   # Client Test ? should i put decorator on it @unittest ?

    def test_multiple_lr(self):
        cxt = tc.Context()  
        # Reading our practice data 
        X_train,X_test,y_train,y_test=sample_data() 
                            
        cxt.inputs = Dense.load(X_train.shape,
                                tc.F32, X_train.flatten.tolist()) 
        #Question : Would Flatten turn (300,2) into (600,1)?? or it will work ?
        cxt.labels = Dense.load(y_train.shape,
                                tc.F32, y_train.flatten.tolist())
        
        #Question : tc.F32  can be used as float32 to hold my lr value?
        cxt.learningrate = tc.F32(0.1)
        # Question : i will need to minimize it by 1 each loop to count ,
        #            so I should use Number or Dense ?
        cxt.iteration= tc.Number(400)
        # Question : I need to pass these attributes. should i use tc.Map or just casting it by [] ?
        cxt.mlr = MultipleLinearRegression.load([cxt.learningrate,cxt.iteration])
        cxt.mlr.fit([cxt.inputs,cxt.labels])
        response=self.host.post(ENDPOINT,cxt)
        print(response)


    
