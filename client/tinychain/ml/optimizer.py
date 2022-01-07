from abc import abstractmethod
from typing import List

from tinychain.collection.tensor import Tensor, Dense
from tinychain.decorators import closure, post_op
from tinychain.ml import Parameter
from tinychain.ref import After, While
from tinychain.state import Map
from tinychain.value import F32, UInt


class Optimizer(Map):
    @abstractmethod
    def optimize(self, i, gradient, inputs, loss):
        """Update the given `gradient` by computing deltas for the given `loss`."""

    @property
    def lr(self) -> F32:
        return self['lr']


class GradientDescent(Optimizer):
    @classmethod
    def create(cls, lr=F32(0.01)):
        """Create a new `GradientDescent` optimizer with the given `learning_rate`."""

        return cls({"lr": lr})

    def optimize(self, i, param_list: List[Parameter]):
        return [param.value.write(param.value - (param.grad * self.lr)) for param in param_list]


class Adam(Optimizer):
    @classmethod
    def create(cls, beta1=F32(0.9), beta2=F32(0.999), lr=F32(1e-3), eps=F32(1e-8)):
        return cls({'beta1': beta1, 'beta2': beta1, 'lr': lr, 'eps': eps})

    @property
    def beta1(self) -> F32:
        return self['beta1']

    @property
    def beta2(self) -> F32:
        return self['beta2']

    @property
    def eps(self) -> F32:
        return self['eps']

    def optimize(self, i, param_list: List[Parameter]):

        if i == 1:
            self.m = [Parameter(name=p.name, value=Dense.zeros(p.value.shape, F32), grad=p.grad) for p in param_list]
            self.v = [Parameter(name=p.name, value=Dense.zeros(p.value.shape, F32), grad=p.grad) for p in param_list]

        update_m = [m.value.write(m.value * self.beta1 + p.grad * (F32(1) - self.beta1)) for m, p in zip(self.m, param_list)]
        update_v = [v.value.write(v.value * self.beta2 + p.grad.pow(2) * (F32(1) - self.beta2)) for v, p in zip(self.v, param_list)]
        a = self.lr * (F32(1) - self.beta2.pow(i)).pow(F32(0.5)) / (F32(1) - self.beta1.pow(i))
        return After(
            when=[update_m, update_v],
            then=[
                p.value.write(p.value - m.value/(v.value.pow(F32(0.5).add(self.eps)))*a)
                for m, v, p in zip(self.m, self.v, param_list)
            ])



def train(model, optimizer, inputs, cost, train_while):
    """
    Train a :class:`Differentiable` such as a neural network while the given `train_while` condition is `True`.

    Two named states are provided to `train_while`:
        `i`: the iteration number, a one-indexed `UInt`
        `output`: a `Tensor`, the last output of the model.
    """

    @closure
    @post_op
    def step(i: UInt, output: Tensor):
        loss = cost(output)
        update = optimizer.optimize(i, model, inputs, loss)
        return After(update, {"i": i + 1, "output": model.forward(inputs).copy()})

    output = model.forward(inputs).copy()
    return While(train_while, step, {"i": 1, "output": output})
