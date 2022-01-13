from abc import abstractmethod

from tinychain.collection.tensor import Tensor
from tinychain.decorators import closure, post_op
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
                return gradient.write(gradient - (deltas * lr))

        # discard the loss here since there's nowhere to propagate it to
        _loss, deltas = gradient.backward(inputs, loss)
        return update(gradient, deltas)


def train(model, optimizer, inputs, labels, cost, train_while):
    """
    Train a :class:`Differentiable` such as a neural network while the given `train_while` condition is `True`.

    Two named states are provided to `train_while`:
        `i`: the iteration number, a one-indexed `UInt`
        `output`: a `Tensor`, the last output of the model.
    """

    @closure(model, optimizer, inputs, labels)
    @post_op
    def step(i: UInt, output: Tensor):
        loss = cost(output, labels)
        update = optimizer.optimize(i, model, inputs, loss)
        return After(update, {"i": i + 1, "output": model.forward(inputs).copy()})

    output = model.forward(inputs).copy()
    return While(train_while, step, {"i": 1, "output": output})
