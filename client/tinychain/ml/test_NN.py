import math
import numpy as np
from typing import List


NUM_EXAMPLES = 10
LEARNING_RATE = 0.01

class RGD:

    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    def optimize(self, i, param_list: List):
        for i in range(len(param_list)):
            param_list[i]['weights'] -= param_list[i]['dW'] * self.lr
            param_list[i]['bias'] -= param_list[i]['db'] * self.lr

class Sigmoid:
    def forward(self, x):
        return 1 / (1 - np.exp(-x))

    def backward(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)


class Layer:
    def __init__(self, i, o, activation=Sigmoid()):
        self.weights = np.random.random([i, o])
        self.bias = np.random.random([o])
        self.activation = activation

    def forward(self, x):
        inputs = np.einsum("ki,ij->kj", x, self.weights) + self.bias
        return self.activation.forward(inputs)

    def backward(self, x, loss):
        m = x.shape[0]

        inputs = np.einsum("ki,ij->kj", x, self.weights) + self.bias
        delta = self.activation.backward(inputs)*loss
        update = np.einsum("ki,kj->ij", x, delta).copy()
        return delta, {'weights': self.weights, 'bias': self.bias, "dW": update / m, "db": delta.sum(0) / m}


class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        state = inputs
        for layer in self.layers:
            state = layer.forward(state)

        return state

    def backward(self, inputs, loss):
        n = len(self.layers)
        layer_inputs = [inputs]

        for layer in self.layers:
            layer_output = layer.forward(layer_inputs[-1]).copy()
            layer_inputs.append(layer_output)
        
        param_list = []
        for l in reversed(range(n)):
            print('_'*10)
            loss, par = self.layers[l].backward(layer_inputs[l], loss)
            print(par)
            loss = loss.copy()
            param_list.extend(par)
        
        return param_list


if __name__ == "__main__":
    inputs = np.random.random([NUM_EXAMPLES, 5]) * 2
    labels = (np.logical_xor(inputs[:, 0] > 1, inputs[:, 1] > 1))
    labels = labels.reshape([NUM_EXAMPLES, 1]).astype(np.float32)

    net = Net([Layer(5, 3), Layer(3, 2), Layer(2, 1)])

    output = net.forward(inputs)
    loss = (output - labels)**2
    dL = (output - labels)*2
    last_loss = int(loss.sum())

    optimizer = RGD(lr=0.1)

    i = 0
    while not ((output > 1) == labels).all():
        param_list = net.backward(inputs, dL)
        optimizer.optimize(param_list)
        print(param_list)
        output = net.forward(inputs)
        if np.isnan(output).any():
            break

        loss = (output - labels)**2

        if i % 1000 == 0:
            print(f"iteration {i}, loss {loss.sum()}")

        i = i + 1

    print(f"{i} iterations")