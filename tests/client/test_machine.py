import numpy as np
import tinychain as tc
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from testutils import ClientTest

Dense = tc.tensor.Dense

ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = tc.F32(0.01)
MAX_ITERATIONS = 10
NUM_EXAMPLES = 20


class Logistic_Regression():
        # ?? no init right ?  so how can it work out ? should i inherit something ? 
    def create (self,inputs):
        layer0= tc.ml.nn.DNNLayer.create('layer0', inputs.shape[1], 1, tc.ml.Sigmoid())
        return tc.ml.nn.Sequential.load([layer0])
    
    def cost(self,output, labels, dl=False):
        if dl:
            return output.sub(labels).mul(2).mean()
        return output.sub(labels).pow(2).mean()
    
    def fit (self,inputs,labels):
        def train_while(i: tc.UInt, output: tc.tensor.Dense):
            return (i <= MAX_ITERATIONS).logical_and(((output > 0.5) != labels).any())
        # for trial only , Adam must be replaced by Gradient Descent !
        optimizer = tc.ml.optimizer.Adam.create(param_list=logistic.get_param_list(), lr=LEARNING_RATE)
        result= tc.ml.optimizer.train(logistic, optimizer, inputs, labels, cost, train_while)
        return result
        
class MLTests(ClientTest):
    def test_logistic(self):
        cxt=tc.Context()
        cancer_dataset=load_breast_cancer()
        x_train,x_test,y_train,y_test=train_test_split(cancer_dataset['data'],cancer_dataset['target'],random_state=0)
        
        cxt.inputs=load(x_train)
        cxt.labels=load(y_train)
        
        cxt.logistic = Logistic_Regression.create(cxt.inputs)
        
        cxt.result=cxt.logistic.fit(cxt.inputs,cxt.labels)
        
        response = self.host.post(ENDPOINT, cxt)
        self.assertEqual(response["i"], MAX_ITERATIONS + 1)


def load(ndarray, dtype=tc.F32):
    return tc.tensor.Dense.load(ndarray.shape, dtype, ndarray.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
