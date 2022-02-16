#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:59:54 2022
@author: meracoda
"""

from tinychain.collection.tensor import einsum, Dense , Tensor
from tinychain.ml import Layer, NeuralNet
from tinychain.ref import After
import tinychain as tc 
import numpy as np


class MultipleLinearRegression :
    
    @classmethod
    def load (self,learningrate): 
        return MultipleLinearRegression([learningrate])
    
    @property
    def learningrate(self):  
        '''
        We use property accessor so we recall self.learningrate instead of Dense(self[0])
        '''
        return Dense(self[0])
    
    def predict(self, inputs, weights, learning = True):
        if learning==True:
            preds = einsum("ij,ki->kj", [weights, inputs]) 
        else:
            #inputs = np.insert(inputs ,0,1,axis=1)   
            inputs_=Tensor.concatenate(1,inputs,axis=1) #Question: i try to add column of ones ! 
            preds = einsum("ij,ki->kj", [weights, inputs_])            
        return preds
    
    def cost_func( self, labels ,preds):#Question: now would it work ?
        cost = Tensor.sum((labels - preds)**2)/labels.len()
        return cost

    def gradient_descent(self , inputs , weights, labels):        
        d_weights = 2/inputs.shape[0]*(einsum("ij,ki->kj", weights, einsum("ij,ki->kj", inputs, inputs.transpose()))-
                                                      einsum("ij,ki->kj", labels, inputs.transpose()))        
        return d_weights


    def update_params(self ,weights, d_weights): 
        new_weights = weights - self.learningrate * d_weights
        return new_weights

    def fit(self, X_train, y_train):
        X_train_ = Tensor.concatenate(1,X_train,axis=1) 
        y_train_ = y_train
        n_samples, n_features = X_train_.shape[1],X_train_.shape[0]
        
        weights = np.zeros((n_features + 1, 1)) #Q: Can i create zeros tc tensor  or i have to create it in numpy then convert?
        weights_= Tensor.load(weights.shape,tc.F32,weights.flatten().tolist())        
        
 
        @tc.get_op
        def loop(until: tc.Number) -> tc.Int:
            @tc.closure
            @tc.post_op
            def cond(i: tc.Int):
                return i < until
        ​
            @tc.post_op
            def step(i: tc.Int) -> tc.Int:
  
                preds=self.predict(X_train_,weights_)
                cost_=self.cost_func(y_train_,preds)

                d_weights=gradient_descent(X_train_,weights_,y_train_)
                weights_=self.update_params(weights_,d_weights)
                
                return tc.Map(i=i + 1 ,cost_)  
        ​
            initial_state = tc.Map(i=0)  
        ​
            # return the loop itself
            return tc.While(cond, step, initial_state)
            

    def R2_Score(self, x, y):
        ss_res = Tensor.sum((self.predict(x, learning = False) - y)**2)
        ss_tot = Tensor.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot
        return r2   
    
    
    
