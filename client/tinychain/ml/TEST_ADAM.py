import math
import numpy as np
import operator
import tinychain as tc
import unittest

from functools import reduce
  
import matplotlib.pyplot as plt
import networkx as nx
import tinychain as tc


def visualize(state, cxt=tc.Context()):
    G = nx.Graph()
    add_edges(G, state, cxt)
    nx.draw_networkx(G)
    plt.show()


def add_edges(graph, state, cxt):
    for dep in tc.util.debug(state):
        if isinstance(dep, tc.URI):
            if dep.id() is not None:
                if dep.id() in cxt.form:
                    dep = cxt.form[dep.id()]
                else:
                    print(f"{dep.id()} not in {cxt}")

        graph.add_edge(repr(state), repr(dep))

        if isinstance(dep, tc.Context):
            cxt = dep
            if cxt.form:
                dep = cxt.form[next(reversed(cxt.form))]
                graph.add_edge(repr(cxt), repr(dep))
                add_edges(graph, dep, cxt)
        else:
            add_edges(graph, dep, cxt)


Dense = tc.tensor.Dense

HOST = tc.host.Host("http://127.0.0.1:8702")
ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = tc.F32(0.01)
MAX_ITERATIONS = 500
NUM_EXAMPLES = 10


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
    
def testAdam():

    # def BCELoss(output, dL=False):
    #     if dL:
    #         return -((output - cxt.labels) / (output - output * cxt.labels)).sum() / output.shape[0]
    #     return -(cxt.labels*output.log() - (cxt.labels.add(-1)) * (output.add(-1)*-1).log()).sum() / output.shape[0]

    def cost(output, dL=False):
        if dL:
            return (output - cxt.labels)*2
        return (output - cxt.labels)**2

    @tc.closure
    @tc.post_op
    def train_while(i: tc.UInt, loss: tc.tensor.Tensor):
        return (i <= MAX_ITERATIONS).logical_and(loss >= 1e-3)

    cxt = tc.Context()

    inputs = np.random.random(NUM_EXAMPLES * 5).reshape([NUM_EXAMPLES, 5])
    labels = np.logical_xor(inputs[:, 0] > 0.5, inputs[:, 1] > 0.5).reshape([NUM_EXAMPLES, 1]).astype(np.float32)

    cxt.inputs = load(inputs)
    cxt.labels = load(labels, tc.F32)

    cxt.input_layer0 = create_layer('layer0', 5, 4, tc.ml.Sigmoid())
    cxt.input_layer1 = create_layer('layer1', 4, 2, tc.ml.ReLU())
    cxt.output_layer = create_layer('layer2', 2, 1, tc.ml.Sigmoid())

    cxt.nn = tc.ml.dnn.DNN.load([cxt.input_layer0, cxt.input_layer1, cxt.output_layer])
    param_list = cxt.nn.get_param_list()
    cxt.optimizer = tc.ml.optimizer.Adam.create(param_list=param_list)
    result = visualize(tc.ml.optimizer.train(cxt.nn, cxt.optimizer, cxt.inputs, cost, train_while))
    print(result)
    # cxt.output = cxt.nn.forward(cxt.inputs)
    # cxt.loss = cost(cxt.output)
    # cxt.dl = cost(cxt.output, dL=True)
    # cxt.result = cxt.nn.backward(cxt.inputs, cxt.dl)

    response = HOST.post(ENDPOINT, cxt)

    print(response)

def load(ndarray, dtype=tc.F32):
    return tc.tensor.Dense.load(ndarray.shape, dtype, ndarray.flatten().tolist())


testAdam()
