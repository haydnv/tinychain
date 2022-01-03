import collections

from abc import abstractmethod

from tinychain.collection.tensor import Dense
from tinychain.decorators import closure, post_op
from tinychain.ml import EPS
from tinychain.ref import While
from tinychain.state import Map
from tinychain.value import F32, UInt


class Optimizer(Map):
    @abstractmethod
    def optimize(self, i, gradients):
        """
        Compute deltas for the given `gradients` using this `Optimizer`.

        `gradients` can be a `Tensor` or a Python `Mapping` or `Iterable`.
        """


class GradientDescent(Optimizer):
    @classmethod
    def create(cls, learning_rate=F32(0.01)):
        """Create a new `GradientDescent` optimizer with the given `learning_rate`."""

        return cls({"lr": learning_rate})

    def optimize(self, i, gradients):
        if isinstance(gradients, collections.abc.Mapping):
            return {name: self.optimize(i, gradients[name]) for name in gradients}
        elif isinstance(gradients, collections.abc.Iterable):
            return [self.optimize(i, gradient) for gradient in gradients]
        else:
            return gradients * self["lr"]


def train(model, optimizer, inputs, cost, max_iterations):
    """Train a :class:`Gradient` such as a neural network in up to `max_iterations` steps."""

    @closure
    @post_op
    def while_cond(i: UInt, loss: Dense):
        fit = (loss > EPS).any()
        return fit.logical_and(i < max_iterations)

    @closure
    @post_op
    def step(i: UInt, loss: Dense):
        loss = model.train(i, inputs, loss, optimizer)
        return {"i": i + 1, "loss": loss}

    loss = cost(output=model.eval(inputs))
    return While(while_cond, step, {"i": 1, "loss": model.train(0, inputs, loss, optimizer)})
