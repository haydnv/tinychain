from abc import abstractmethod, ABC
from tinychain.collection.tensor import einsum, Dense
from tinychain.decorators import post_method
from tinychain.ref import After, MethodSubject, Put
from tinychain.state import Tuple
from tinychain.util import URI


class Activation(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class Sigmoid(Activation):
    def forward(self, Z):
        return 1 / (1 + (-Z).exp())

    def backward(self, dA, Z):
        sig = self.forward(Z=Z)
        return dA * sig * (1 - sig)


class ReLU(Activation):
    def forward(self, Z):
        return Z * (Z > 0)

    def backward(self, dA, Z):
        return dA * (Z > 0)


def layer(weights, bias, activation):
    class Layer(Tuple):
        def eval(self, inputs):
            return activation.forward(einsum("ij,ki->kj", [self[0], inputs]) + self[1])

        def gradients(self, A_prev, dA, Z, m):
            dZ = activation.backward(dA, Z).copy()

            d_weights = einsum("kj,ki->ij", [dZ, A_prev]) / m
            d_bias = dZ.sum(0) / m

            dA_prev = einsum("ij,kj->kj", [self[0], dZ])
            return dA_prev, d_weights, d_bias

        def train_eval(self, inputs):
            Z = einsum("ij,ki->kj", [self[0], inputs]) + self[1]
            A = activation.forward(Z)
            return A, Z

        def update(self, d_weights, d_bias):
            new_weights = weights - d_weights
            new_bias = bias - d_bias
            return Dense(self[0]).overwrite(new_weights), Dense(self[1]).overwrite(new_bias)

    return Layer([weights, bias])


def neural_net(layers, learning_rate):
    num_layers = len(layers)

    class NeuralNet(Tuple):
        def eval(self, inputs):
            state = layers[0].eval(inputs)
            for i in range(1, len(layers)):
                state = layers[i].eval(state)

            return state

        def train(self, inputs, labels):
            A = [inputs]
            Z = [None]

            for layer in layers:
                A_l, Z_l = layer.train_eval(A[-1])
                A.append(A_l)
                Z.append(Z_l)

            labels = labels.expand_dims().copy()
            dA = (labels / A[-1]) - ((1 - labels) / (1 - A[-1]))

            updates = []
            for i in reversed(range(0, num_layers)):
                dA, d_weights, d_bias = layers[i].gradients(A[i], dA, Z[i + 1], num_layers)
                update = layers[i].update(d_weights * learning_rate, d_bias * learning_rate)
                updates.append(update)

            return updates

    return NeuralNet(layers)
