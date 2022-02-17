


import math

import numpy as np
import tinychain as tc
import unittest
from tinychain.collection.tensor import einsum, Dense , Tensor


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
    y_train : np.array
        Labels to be trained on.
    y_test : np.array
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
    
@unittest    
class MLTests(ClientTest):   

    def test_multiple_lr(self):
        cxt = tc.Context()  
        # Reading our practice data 
        X_train,X_test,y_train,y_test=sample_data() 
                            
        cxt.X_train = Dense.load(X_train.shape,tc.F32, X_train.flatten.tolist()) 
        cxt.X_test = Dense.load(X_test.shape,tc.F32, X_test.flatten.tolist()) 
        cxt.y_train = Dense.load(y_train.shape,tc.F32, y_train.flatten.tolist()) 
        cxt.y_test = Dense.load(y_test.shape,tc.F32, y_test.flatten.tolist())        
        
        cxt.learningrate = tc.Number(0.1)
        cxt.iteration= tc.Number(400)
        cxt.mlr = MultipleLinearRegression.load(cxt.learningrate)       
        cxt.weights= cxt.mlr.fit([cxt.X_train,cxt.y_train,cxt.iteration])
        cxt.preds=cxt.mlr.predict(cxt.X_test ,learning=False)
        
        cxt.score=cxt.mlr.R2_Score(cxt.X_test,cxt.y_test)
        self.assertGreaterEqual(cxt.score,0.9)
               
        
        response=self.host.post(ENDPOINT,cxt)
        print(response)



if __name__ == "__main__":
    unittest.main()    
