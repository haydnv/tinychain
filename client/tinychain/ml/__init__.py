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

    return Layer(weights)


def neural_net(layers):
    class NeuralNet(Tuple):
        def eval(self, inputs):
            state = layers[0].eval(inputs)
            for i in range(1, len(layers)):
                state = layers[i].eval(state)

            return state

    return NeuralNet(layers)
