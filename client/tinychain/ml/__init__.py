from abc import abstractmethod, ABC

from tinychain.state import Map, Tuple

EPS = 10**-6


class Activation(ABC):
    """A differentiable activation function for a neural network."""

    @abstractmethod
    def forward(self, Z):
        """Compute the activation of the given `Tensor`"""

    @abstractmethod
    def backward(self, dA, Z):
        """Compute the partial differential of this function with respect to the given loss"""


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


class Gradient(object):
    @classmethod
    @abstractmethod
    def shape(cls):
        """
        Return the shape of this gradient.

        This can be a `Tuple` of `U64` dimensions (for a `Tensor`) or a Python `list` or `dict`
        (for a more complex trainable data structure like a `NeuralNet`).
        """

    @abstractmethod
    def eval(self, inputs):
        """Evaluate this `Gradient` with respect to the given `inputs`."""

    @abstractmethod
    def train(self, i, inputs, loss, optimizer):
        """Update this `Gradient` with respect to the given `inputs` and `loss` using the given `optimizer`."""


class Layer(Map, Gradient):
    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass


class NeuralNet(Tuple, Gradient):
    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass
