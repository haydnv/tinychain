from abc import abstractmethod
import typing as t

import tinychain as tc
from tinychain.collection.tensor import Tensor, Dense
from tinychain.decorators import closure, post_op
from tinychain.ref import After, While
from tinychain.state import Map
from tinychain.value import F32, UInt
from tinychain.ml import EPS, Parameter, DiffedParameter

from client.tinychain.ml import Parameter, DiffedParameter


class Optimizer(Map):
    @abstractmethod
    def optimize(self):
        """Update the given `gradient` by computing deltas for the given `loss`."""

    @property
    def lr(self) -> F32:
        return self['lr']


class GradientDescent(Optimizer):
    @classmethod
    def create(cls, lr=F32(0.01)):
        """Create a new `GradientDescent` optimizer with the given hyperparameter.
        'lr': learning rate
        """

        return cls({"lr": lr})

    def optimize(self, i, param_list: t.List[Parameter]):
        return [param.value.write(param.value - (param.grad * self.lr)) for param in param_list]

class Adam(Optimizer):

    @classmethod
    def create(cls, param_list: t.List[Parameter],  beta1=F32(0.9), beta2=F32(0.999), lr=F32(1e-3), eps=F32(1e-8)):
        """Create a new `Adam` optimizer with the given hyperparameters.
        'lr': learning rate;
        'beta1': The exponential decay rate for the 1st moment estimates;
        'beta2': The exponential decay rate for the 2nd moment estimates;
        'eps': A small constant for numerical stability.
        Args:
            `param_list`: a `List[Parameter]` of model's parameters for optimizing.
        """

        m = {p.name: Dense.zeros(p.value.shape, F32) for p in param_list}
        v = {p.name: Dense.zeros(p.value.shape, F32) for p in param_list}


        class _Adam(cls):

            def optimize(self, i, param_list: t.List[DiffedParameter]):
                update_m = [m[p.name].write(m[p.name] * beta1 + p.grad * (F32(1) - beta1)) for p in param_list]
                update_v = [v[p.name].write(v[p.name] * beta2 + p.grad.pow(2) * (F32(1) - beta2)) for p in param_list]
                a = lr * (F32(1) - beta2.pow(i)).pow(F32(0.5)) / (F32(1) - beta1.pow(i))
                return After(
                    when=[update_m, update_v],
                    then=[
                        p.value.write(p.value - m[p.name] / (v[p.name].pow(F32(0.5)).add(eps)) * a)
                        for p in param_list
                    ])

            @property
            def lr(self) -> F32:
                return lr

        return _Adam()


def train(model, optimizer, inputs, labels, cost, train_while):
    """
    Train a :class:`Differentiable` such as a neural network while the given `train_while` condition is `True`.

    Two named states are provided to `train_while`:
        `i`: the iteration number, a one-indexed `UInt`;
        `output`: a `Tensor`, the last output of the model;
        `loss`: a `Number`, the lats loss of the model's predict.
    """

    @closure(model, optimizer, inputs, labels)
    @post_op
    def step(i: UInt, output: Tensor):
        loss = cost(output, labels)
        update = optimizer.optimize(i, model, inputs, loss)
        return After(update, {"i": i + 1, "output": model.forward(inputs).copy()})

    output = model.forward(inputs).copy()
    loss = cost(output)
    return While(train_while, step, {"i": 1, "output": output, "loss": loss})
