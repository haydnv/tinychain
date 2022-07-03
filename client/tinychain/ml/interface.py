import typing

from .. import error
from ..collection.tensor import Tensor
from ..decorators import differentiable, post
from ..interface import Interface
from ..math.interface import Numeric
from ..scalar.value import Id
from ..state import State


class Gradient(State, Numeric):
    """Helper class to handle the case of either a :class:`Number` or :class:`Tensor` as a gradient"""


Gradients = typing.Dict[Id, Gradient]


class Differentiable(Interface):
    """A :class:`Differentiable` state in an ML model, which can be used in the composition of a more complex model"""

    @differentiable
    def eval(self, inputs: Tensor) -> Tensor:
        """Evaluate this :class:`Differentiable` with respect to the given `inputs`."""

        return error.NotImplemented(f"Differentiable.eval for {self.__class__.__name__}")

    @post
    def gradient(self, inputs: Tensor, loss: Tensor) -> Gradient:
        """Return a :class:`Map` of gradients per member :class:`Variable` of this :class:`Differentiable` state."""

        return error.NotImplemented(f"Differentiable.gradient for {self.__class__.__name__}")
