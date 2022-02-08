#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:59:54 2022
@author: meracoda
"""

from tinychain.collection.tensor import einsum, Dense
from tinychain.ml import Layer, NeuralNet
from tinychain.ref import After
import numpy as np


class MultipleLinearRegression :
    
    @property
    def learningrate(self):  
        '''
        We use property accessor so we recall self.learningrate instead of Dense(self[0])
        '''
        return Dense(self[0])
    @property
    def iteration(self):  
        '''
        We use property accessor so we recall self.iteration instead of Dense(self[1])
        '''
        return Dense(self[1])
    
    def load (self):  # Question  i need to instanitiate my class with this 
                                    # i guess i need to add @class_method ?
        return MultipleLinearRegression([learningrate,iteration])
    
    def predict(self, inputs, weights, learning = True):
        if learning==True:
            preds = tc.tensor.einsum("ij,ki->kj", [weights, inputs]) 
        else:
            inputs = np.insert(inputs ,0,1,axis=1)   # i want to add column of ones at index 0
            # inputs = tc.concatenate(inputs,0 , axis =1) how about this ?
            preds = tc.tensor.einsum("ij,ki->kj", [weights, inputs])            
        return preds
    
    def cost_func( self, labels ,preds):
        # is it okay to use numpy.sum ?
        cost = np.sum((labels - preds)**2)/len(labels)
        return cost

    def gradient_descent(self , inputs , weights, labels):
        # Still trying to convert this 
        d_weights = 2/len(inputs)*(inputs.T.dot(inputs).dot(weights) - inputs.T.dot(labels))  
        
        return d_weights


    def update_params(self ,weights, d_weights): 
        new_weights = weights - self.learningrate * d_weights
        return new_weights

# Rest not converted it .......................................    
    def fit(self, X_train, y_train):         
        X_train = np.insert(X_train,0,1,axis=1)
        y_train = y_train
        n_samples, n_features = X_train.shape[1],X_train.shape[0]

        weights = np.zeros((n_features + 1, 1))
        
        while(self.itr+1):
            costs = []    
            self.predict()
            self.cost_func()
            costs.append(self.cost)
            self.gradient_descent()
            self.update_params()
            self.itr -= 1
        return costs

    def R2_Score(self, x, y):
        ss_res = sum((self.predict(x, learning = False) - y)**2)
        ss_tot = sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot
        return r2   
    
    
    
