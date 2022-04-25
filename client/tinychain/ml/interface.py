from .. import error
from ..collection.tensor import Tensor
from ..decorators import differentiable
from ..interface import Interface


class Differentiable(Interface):
    """A :class:`Differentiable` state in an ML model, which can be used in the composition of a more complex model"""

    @differentiable
    def eval(self, inputs: Tensor) -> Tensor:
        """Evaluate this :class:`Differentiable` with respect to the given `inputs`."""

        return error.NotImplemented(f"Differentiable interface for {self.__class__.__name__}")
