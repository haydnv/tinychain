from abc import abstractmethod

from tinychain.collection.tensor import Dense
from tinychain.decorators import closure, post_op
from tinychain.ml import EPS
from tinychain.ref import After, While
from tinychain.state import Map
from tinychain.value import F32, UInt


class Optimizer(Map):
    @abstractmethod
    def optimize(self, i, gradient, inputs, loss):
        """Update the given `gradient` by computing deltas for the given `loss`."""


class GradientDescent(Optimizer):
    @classmethod
    def create(cls, learning_rate=F32(0.01)):
        """Create a new `GradientDescent` optimizer with the given `learning_rate`."""

        return cls({"lr": learning_rate})

    def optimize(self, i, gradient, inputs, loss):
        lr = self["lr"]

        def update(gradient, deltas):
            shape = gradient.shape

            if isinstance(shape, dict):
                return {name: update(gradient[name], deltas[name]) for name in shape}
            elif isinstance(shape, list) or isinstance(shape, tuple):
                return [update(gradient[n], deltas[n]) for n in range(len(shape))]
            else:
                return gradient.write(deltas * lr)

        loss, deltas = gradient.backward(inputs, loss)
        return After(update(gradient, deltas), loss)


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
        loss = optimizer.optimize(i, model, inputs, loss)
        return {"i": i + 1, "loss": loss}

    return While(while_cond, step, {"i": 0, "loss": cost(model.forward(inputs))})
