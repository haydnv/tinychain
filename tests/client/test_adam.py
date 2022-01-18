import json
import math

import numpy as np
import tinychain as tc
from tinychain.collection.tensor import Tensor

np.random.seed(2)

# TODO: implement AdamOptimizer
def truncated_normal(size, mean=0., std=None):
    std = std if std else math.sqrt(size)

    while True:
        dist = np.random.normal(mean, std, size)
        truncate = np.abs(dist) > mean + (std * 2)
        if truncate.any():
            new_dist = np.random.normal(mean, std, size) * truncate
            dist *= np.logical_not(truncate)
            dist += new_dist
        else:
            return dist

def create_layer(name, input_size, output_size, activation):
    shape = (input_size, output_size)
    bias = tc.tensor.Dense.load([output_size], tc.F32, truncated_normal(output_size).tolist())
    weights = tc.tensor.Dense.load(shape, tc.F32, truncated_normal(input_size * output_size).tolist())
    return tc.ml.dnn.DNNLayer.load(name, weights, bias, activation)

def load(ndarray, dtype=tc.F32):
    return tc.tensor.Dense.load(ndarray.shape, dtype, ndarray.flatten().tolist())

Dense = tc.tensor.Dense

HOST = tc.host.Host("http://127.0.0.1:8702")
ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = tc.F32(0.01)
MAX_ITERATIONS = 25
NUM_EXAMPLES = 10


#Test Adam step by step
def testAdam_h():

    def cost(output, labels, dl=False):
        if dl:
            return output.sub(labels).mul(2)
        return output.sub(labels).pow(2)
    
    cxt = tc.Context()

    #generate inputs
    inputs = np.random.random(NUM_EXAMPLES * 2).reshape([NUM_EXAMPLES, 2])
    cxt.inputs = load(inputs)

    #calculate labels as y= x1*0.6 + x2*0.2 + 0.15
    labels = (inputs[:, 0]*0.6 + inputs[:, 1]*0.2 + 0.15).astype(np.float32).reshape([2, 1])
    cxt.labels = load(labels)

    #create DNN
    cxt.input_layer0 = create_layer('layer0', 2, 3, tc.ml.Sigmoid())
    cxt.input_layer1 = create_layer('layer1', 3, 2, tc.ml.Sigmoid())
    cxt.output_layer = create_layer('layer2', 2, 1, tc.ml.Sigmoid())
    cxt.nn = tc.ml.dnn.DNN.load([
        cxt.input_layer0,
        cxt.input_layer1,
        cxt.output_layer])

    #get param_list for optimizer
    param_list = cxt.nn.get_param_list()

    #create Adam optimizer with (beta1=0.9, beta2=0.999, lr=0.01, eps=1e-8)
    cxt.optimizer = tc.ml.optimizer.Adam.create(param_list=param_list, lr=tc.F32(LEARNING_RATE))

    #train DNN model
    #step1
    cxt.output = cxt.nn.forward(cxt.inputs)
    cxt.out_0 = cxt.output.copy()

    #step2
    cxt.loss = cost(cxt.out_0, cxt.labels)
    cxt.dl = cost(cxt.output, cxt.labels, dl=True)
    cxt.out_1 = Tensor(tc.After(cxt.optimizer.optimize(1, cxt.nn.backward(cxt.inputs, cxt.dl)), cxt.nn.forward(cxt.inputs)))
    cxt.o1 = cxt.out_1.copy()

    #step3
    cxt.loss_1 = cost(cxt.out_1, cxt.labels)
    cxt.dl_1 = cost(cxt.out_1, cxt.labels, dl=True)
    cxt.out_2 = Tensor(tc.After(cxt.optimizer.optimize(2, cxt.nn.backward(cxt.inputs, cxt.dl_1)), cxt.nn.forward(cxt.inputs))).copy()

    #result
    cxt.result = tc.Map(o0=cxt.out_0, o1=cxt.out_1, o2=cxt.out_2)

    with open('adam_dump_step_by_step.json','w') as f:
        json.dump(tc.to_json(cxt), f, indent=4)
    
    response = HOST.post(ENDPOINT, cxt)
    print(response)

#Test Adam with optimizer.train
def testAdam_a():

    def cost(output, labels, dl=False):
        if dl:
            return output.sub(labels).mul(2).sum() / output.shape[0]
        return output.sub(labels).pow(2).sum() / output.shape[0]
    
    cxt = tc.Context()

    #generate inputs
    inputs = np.random.random(NUM_EXAMPLES * 4).reshape([NUM_EXAMPLES, 4])
    #cxt.inputs = load(inputs)
    cxt.output = load(np.array([1, 1, 1, 1, 1]).reshape(5, 1))
    #calculate labels as y= x1*0.6 + x2*0.2 + 0.15
    labels = (inputs[:, 0] > 0.5).astype(np.float32).reshape([NUM_EXAMPLES, 1])
    cxt.labels = load(np.array([[0.], [0.], [0.], [1.], [0.]]))
    cxt.res = ((cxt.output > 0.5) != cxt.labels).any()#cxt.output.sub(cxt.labels).pow(2).sum() / cxt.output.shape[0]
    # #create DNN
    # cxt.input_layer0 = create_layer('layer0', 4, 3, tc.ml.Sigmoid())
    # cxt.input_layer1 = create_layer('layer1', 3, 2, tc.ml.ReLU())
    # cxt.output_layer = create_layer('layer2', 2, 1, tc.ml.Sigmoid())
    # cxt.nn = tc.ml.dnn.DNN.load([
    #     cxt.input_layer0,
    #     cxt.input_layer1,
    #     cxt.output_layer])

    # #create Adam optimizer with (beta1=0.9, beta2=0.999, lr=0.01, eps=1e-8)
    # cxt.optimizer = tc.ml.optimizer.Adam.create(param_list=cxt.nn.get_param_list(), lr=LEARNING_RATE)

    # #train model
    # @tc.closure(cxt.labels)
    # @tc.post_op
    # def train_while(i: tc.UInt, output: tc.tensor.Tensor):
    #     return (i <= MAX_ITERATIONS).logical_and((output.sub(cxt.labels).pow(2).sum() >= 1e-6))

    # cxt.result = tc.ml.optimizer.train(cxt.nn, cxt.optimizer, cxt.inputs, cxt.labels, cost, train_while)
    response = HOST.post(ENDPOINT, cxt)
    print(response)

#testAdam_h()
testAdam_a()