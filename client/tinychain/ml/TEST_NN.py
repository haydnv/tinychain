import math
import numpy as np


NUM_EXAMPLES = 100
LEARNING_RATE = 0.1


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
        inputs = np.matmul(x, self.weights) + self.bias
        return self.activation.forward(inputs)

    def backward(self, x, loss):
        m = x.shape[0]
        inputs_to_act = np.matmul(x, self.weights) + self.bias
        delta = loss * self.activation.backward(inputs_to_act)
        p = delta@self.weights.T
        delta_w = np.matmul(x.T, delta)
        self.weights -= delta_w * LEARNING_RATE

        delta_b = (self.bias * delta).sum(0)
        self.bias -= delta_b * LEARNING_RATE
        return p


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

        for lb in reversed(range(n)):
            loss = self.layers[lb].backward(layer_inputs[lb], loss).copy()


if __name__ == "__main__":
    inputs = np.random.random([NUM_EXAMPLES, 5]) * 2
    labels = (np.logical_xor(inputs[:, 0] > 1, inputs[:, 1] > 1))
    labels = labels.reshape([NUM_EXAMPLES, 1]).astype(np.float)

    net = Net([Layer(5, 3), Layer(3, 2), Layer(2, 1)])

    output = net.forward(inputs)
    loss = (output - labels)**2
    last_loss = int(loss.sum())

    i = 0
    while not ((output > 1) == labels).all():
        net.backward(inputs, loss)
        output = net.forward(inputs)
        if np.isnan(output).any():
            break

        loss = (output - labels)**2

        if i % 1000 == 0:
            print(f"iteration {i}, loss {loss.sum()}")

        i = i + 1

    print(f"{i} iterations")