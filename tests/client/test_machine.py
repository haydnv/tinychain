import numpy as np
import tinychain as tc
import unittest
from sklearn.model_selection import train_test_split
from testutils import ClientTest

Dense = tc.tensor.Dense
ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = tc.F32(0.01)
MAX_ITERATIONS = 10
NUM_EXAMPLES = 20


class Logistic_Regression():
    def fit (self,inputs ,labels):
        layer0= tc.ml.nn.DNNLayer.create('layer0', inputs.shape[1], 1, tc.ml.Sigmoid())
        logistic = tc.ml.nn.Sequential.load([layer0])
        def train_while(i: tc.UInt, output: tc.tensor.Dense):
            return (i <= MAX_ITERATIONS).logical_and(((output > 0.5) != labels).any())
        
        def cost(self,output, labels, dl=False):
            if dl:
                return output.sub(labels).mul(2).mean()
            return output.sub(labels).pow(2).mean()        
        
        # for trial only , Adam must be replaced by Gradient Descent !
        optimizer = tc.ml.optimizer.Adam.create(param_list=logistic.get_param_list(), lr=LEARNING_RATE)
        result= tc.ml.optimizer.train(logistic, optimizer, inputs, labels, cost, train_while)
        return result
        
class MLTests(ClientTest):
    def test_logistic(self):
        cxt=tc.Context()
        data= np.array([[11,-2,1],
               [3,-4,0],
               [2,-10,0],
               [10,-5,1],
               [1,-3,0],
               [6,-5,1],
               [2,-7,0],
               [6,-1,1]])
        X=data[:,:-1]
        y=data[:,-1]
        
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)       
        cxt.inputs=load(X_train)
        cxt.labels=load(y_train)        
        cxt.result = Logistic_Regression.fit(cxt.inputs,cxt.labels)                
        response = self.host.post(ENDPOINT, cxt)
        self.assertEqual(response["i"], MAX_ITERATIONS + 1)


def load(ndarray, dtype=tc.F32):
    return tc.tensor.Dense.load(ndarray.shape, dtype, ndarray.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
