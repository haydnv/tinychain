from .. import error
from ..collection.tensor import Tensor
from ..decorators import differentiable, post
from ..generic import Map
from ..interface import Interface
from ..math.interface import Numeric
from ..state import State


class Gradient(State, Numeric):
    """Helper class to handle the case of either a :class:`Number` or :class:`Tensor` as a gradient"""


class Differentiable(Interface):
    """A :class:`Differentiable` state in an ML model, which can be used in the composition of a more complex model"""

    @differentiable
    def eval(self, inputs: Tensor) -> Tensor:
        """Evaluate this :class:`Differentiable` with respect to the given `inputs`."""

        cls = self.instance.__class__ if hasattr(self, "instance") else self.__class__
        return error.NotImplemented(f"Differentiable.eval for {cls.__name__}")

    @post
    def gradient(self, inputs: Tensor, loss: Tensor) -> Map[Gradient]:
        """Return a :class:`Map` of gradients per member :class:`Variable` of this :class:`Differentiable` state."""

        cls = self.instance.__class__ if hasattr(self, "instance") else self.__class__
        return error.NotImplemented(f"Differentiable.gradient for {cls.__name__}")
