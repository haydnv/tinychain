from abc import abstractmethod, ABC

from tinychain.state import Tuple


class Activation(ABC):
    @abstractmethod
    def forward(self, Z):
        pass

    @abstractmethod
    def backward(self, dA, Z):
        pass


class Sigmoid(Activation):
    def forward(self, Z):
        return 1 / (1 + (-Z).exp())

    def backward(self, dA, Z):
        sig = self.forward(Z=Z)
        return sig * (1 - sig) * dA


class ReLU(Activation):
    def forward(self, Z):
        return Z * (Z > 0)

    def backward(self, dA, Z):
        return (Z > 0) * dA


class Layer(Tuple):
    @abstractmethod
    def eval(self, inputs):
        pass


class NeuralNet(Tuple):
    @abstractmethod
    def eval(self, inputs):
        pass

    @abstractmethod
    def train(self, inputs, cost):
        pass
