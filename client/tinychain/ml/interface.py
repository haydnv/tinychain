from .. import error
from ..collection.tensor import Tensor
from ..decorators import hidden, post
from ..interface import Interface


class Differentiable(Interface):
    """A :class:`Differentiable` state in an ML model, which can be used in the composition of a more complex model"""

    @hidden
    def operator(self, inputs):
        return error.NotImplemented(f"{self.__class__.__name__}")

    @post
    def eval(self, inputs: Tensor) -> Tensor:
        """Evaluate this :class:`Differentiable` with respect to the given `inputs`."""

        return self.operator(inputs)
