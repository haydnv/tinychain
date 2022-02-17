

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
        We use property accessor so we recall self.learningrate instead of tc.Number(self[0])
        '''
        return tc.Number(self[0])
    
    def predict(self, inputs, weights, learning = True):
        if learning==True:
            preds = einsum("ij,ki->kj", [weights, inputs]) 
        else:
            '''
            I try to add column of ones ! 
            '''
 
            inputs_=Tensor.concatenate(1,inputs,axis=1) 
            preds = einsum("ij,ki->kj", [weights, inputs_])            
        return preds
    
    def cost_func( self, labels ,preds):
        '''
        I try to apply Mean Squared Error
        '''
        cost = Tensor.sum((labels - preds)**2)/labels.len()
        return cost

    def gradient_descent(self , inputs , weights, labels):        
        d_weights = 2/inputs.shape[0]*(einsum("ij,ki->kj", weights, einsum("ij,ki->kj", inputs, inputs.transpose()))-
                                                      einsum("ij,ki->kj", labels, inputs.transpose()))        
        return d_weights


    def update_params(self ,weights, d_weights): 
        new_weights = weights - self.learningrate * d_weights
        return new_weights

    def fit(self, X_train, y_train,iteration):
        X_train_ = Tensor.concatenate(1,X_train,axis=1) 
        y_train_ = y_train
        n_samples, n_features = X_train_.shape[1],X_train_.shape[0]
        '''
        Can i create zeros tc tensor  or i have to create it in numpy then convert?
        '''       
        weights = np.zeros((n_features + 1, 1)) 
        weights_= Tensor.load(weights.shape,tc.F32,weights.flatten().tolist())        
        
 
        @tc.get_op
        def loop(iteration: tc.Number) -> tc.Int:
            @tc.closure
            @tc.post_op
            def cond(i: tc.Int):
                return i < iteration
        ​
            @tc.post_op
            def step(i: tc.Int) -> tc.Int:
  
                preds=self.predict(X_train_,weights_,learning=True)
                cost_=self.cost_func(y_train_,preds)

                d_weights=self.gradient_descent(X_train_,weights_,y_train_)
                weights_=self.update_params(weights_,d_weights)
                
                return tc.Map(i=i + 1 ,cost_,weights_)  
        ​
            initial_state = tc.Map(i=0,cost_=0,weights_=weights_)  
        ​
            # return the loop itself
            return tc.While(cond, step, initial_state)
        return weights_
            

    def R2_Score(self, inputs, labels):
        ss_res = Tensor.sum((self.predict(inputs, learning = False) - y)**2)
        ss_tot = Tensor.sum((labels - Tensor.mean(labels))**2)
        r2 = 1 - ss_res/ss_tot
        return r2   
    
    
    
