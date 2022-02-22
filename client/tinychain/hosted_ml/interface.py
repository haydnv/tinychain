from tinychain.decorators import hidden, post_method
from tinychain.state import Interface
from tinychain import error


class Differentiable(Interface):
    """A :class:`Differentiable` state in an ML model, which can be used in the composition of a more complex model"""

    @post_method
    def forward(self, inputs):
        """Evaluate this :class:`Differentiable` with respect to the given `inputs`."""

        return error.NotImplemented("abstract")

    @post_method
    def backward(self, inputs, loss):
        """
        Compute the gradient of this :class`Differentiable` with respect to the given `inputs` and `loss`.

        Returns a tuple `(loss, diffed_params)` where `loss` is the loss to propagate further backwards and
        `diffed_params` is a flattened list of the :class:`DiffedParameter` s for an :class:`Optimizer` to optimize.
        """

        return error.NotImplemented("abstract")
