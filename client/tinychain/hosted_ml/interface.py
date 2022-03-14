from .. import error
from ..decorators import post_method
from ..interface import Interface


class Differentiable(Interface):
    """A :class:`Differentiable` state in an ML model, which can be used in the composition of a more complex model"""

    @post_method
    def eval(self, inputs):
        """Evaluate this :class:`Differentiable` with respect to the given `inputs`."""

        return error.NotImplemented(f"{self.__class__.__name__}.eval")
