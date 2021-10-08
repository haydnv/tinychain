from abc import abstractmethod, ABC
from tinychain.collection.tensor import einsum
from tinychain.state import Tuple


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
            return activation.forward(einsum("ij,ki->kj", [weights, inputs]) + bias)

        def train_eval(self, inputs):
            Z = einsum("ij,ki->kj", [weights, inputs]) + bias
            A = activation.forward(Z)
            return A, Z

    return Layer(weights)


def neural_net(layers):
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

            error = (A[-1] - labels.expand_dims())**2

            return error

    return NeuralNet(layers)
