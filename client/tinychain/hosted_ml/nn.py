import typing

from tinychain.app import Dynamic, Model
from tinychain.collection.tensor import einsum, Dense, Tensor
from tinychain.decorators import post_method

from .activation import Activation
from .interface import Differentiable
from . import LIB_URI


class Layer(Model, Differentiable):
    """A :class:`Layer` in a :class:`NeuralNet`"""

    __uri__ = LIB_URI + "/Layer"


class DNNLayer(Layer, Dynamic):
    @classmethod
    def create(cls, input_size, output_size, activation=None):
        optimal_std = activation.optimal_std if activation else Activation.optimal_std
        std = optimal_std(input_size, output_size)
        weights = Dense.random_normal([input_size, output_size], std=std)
        bias = Dense.random_normal([output_size], std=std)
        return cls(weights, bias, activation)

    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    @post_method
    def forward(self, inputs: Tensor) -> Tensor:
        x = einsum("ki,ij->kj", [inputs, self.weights]) + self.bias
        if self.activation is None:
            return x
        else:
            return self.activation.forward(inputs=x)


class NeuralNet(Model, Differentiable):
    """A neural network"""

    __uri__ = LIB_URI + "/NeuralNet"


class Sequential(NeuralNet, Dynamic):
    def __init__(self, layers):
        self.layers = layers

    @post_method
    def forward(self, inputs: typing.Tuple[Tensor]) -> Tensor:
        state = self.layers[0].forward(inputs=inputs)
        for i in range(1, len(self.layers)):
            state = self.layers[i].forward(inputs=state)

        return state
