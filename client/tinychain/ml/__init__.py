from abc import abstractmethod, ABC

from tinychain.state import Map, Tuple
from tinychain.collection.tensor import Dense

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


class Differentiable(object):
    @classmethod
    @property
    @abstractmethod
    def shape(cls):
        """
        Return the shape of this gradient.

        This can be a `Tuple` of `U64` dimensions (for a `Tensor`) or a Python `list` or `dict`
        (for a more complex trainable data structure like a `NeuralNet`).
        """

    @abstractmethod
    def forward(self, inputs):
        """Evaluate this `Differentiable` with respect to the given `inputs`."""

    @abstractmethod
    def backward(self, inputs, loss):
        """
        Compute the gradient of this `Differential` with respect to the given `inputs` and `loss`.

        Returns a tuple `(loss, gradient)` where `loss` is the loss to propagate further backwards and `gradient` is
        the total gradient for an `Optimizer` to use in order to calculate an update to this `Differentiable`.
        """

    @abstractmethod
    def write(self, new_values):
        """
        Overwrite the values of this `Gradient` with the given `new_values`.

        `new_values` must have the same shape as this `Gradient`.
        """

        shape = self.shape
        if isinstance(shape, dict):
            return {name: self[name].write(new_values[name]) for name in shape}
        elif isinstance(shape, list) or isinstance(shape, tuple):
            return [self[i].write(new_values[i]) for i in range(len(shape))]
        else:
            raise NotImplementedError(f"{self.__class__} needs a `write` method")


class Layer(Map, Differentiable):
    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass


class NeuralNet(Tuple, Differentiable):
    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass


class Parameter:

    def __init__(self, name: str, value: Dense, grad: Dense):
        self.name = name
        self.value = value
        self.grad = grad
