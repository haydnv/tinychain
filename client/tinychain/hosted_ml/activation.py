from tinychain.app import Model
from tinychain.collection.tensor import Tensor
from tinychain.decorators import hidden, post_method
from tinychain import error

from . import LIB_URI


class Activation(Model):
    """A differentiable activation function for a :class:`Layer`."""

    __uri__ = LIB_URI + "/Activation"

    @post_method
    def forward(self, inputs: Tensor):
        """Compute the activation of the given :class:`Tensor`"""

        return error.NotImplemented("abstract")

    @post_method
    def backward(self, inputs: Tensor):
        """Compute the partial differential of this function"""

        return error.NotImplemented("abstract")

    @staticmethod
    @hidden
    def optimal_std(input_size, output_size):
        """Calculate the optimal initial standard deviation for the inputs to this :class:`Activation`"""

        return (input_size * output_size) ** (-0.5)


class Sigmoid(Activation):
    """Sigmoid activation function"""

    __uri__ = LIB_URI + "/Sigmoid"

    @post_method
    def forward(self, inputs: Tensor) -> Tensor:
        return 1 / (1 + (-inputs).exp())

    @post_method
    def backward(self, inputs: Tensor) -> Tensor:
        sig = self.forward(inputs=inputs)
        return sig * (1 - sig)

    @staticmethod
    @hidden
    def optimal_std(input_size, output_size):
        return 1.0 * (2 / (input_size + output_size))**0.5
