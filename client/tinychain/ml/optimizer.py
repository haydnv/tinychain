from abc import abstractmethod
from typing import List, Dict

from tinychain.collection.tensor import Dense, Tensor
from tinychain.decorators import closure, post_op
from tinychain.ref import After, While
from tinychain.state import Map
from tinychain.value import F32, UInt
from tinychain.ml import EPS, Parameter, DiffedParameter


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
        """Create a new `GradientDescent` optimizer with the given hyperparameter.
        'lr': learning rate
        """

        return cls({"lr": lr})

    def optimize(self, i, param_list: List[Parameter]):
        return [param.value.write(param.value - (param.grad * self.lr)) for param in param_list]


class Adam(Optimizer):
    @classmethod
    def create(cls, param_list: List[Parameter],  beta1=F32(0.9), beta2=F32(0.999), lr=F32(1e-3), eps=F32(1e-8)):
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

        return cls({'beta1': beta1, 'beta2': beta2, 'lr': lr, 'eps': eps, 'm': m, 'v': v})

    @property
    def beta1(self) -> F32:
        return self['beta1']

    @property
    def beta2(self) -> F32:
        return self['beta2']

    @property
    def eps(self) -> F32:
        return self['eps']

    @property
    def m(self) -> Dict[str, Tensor]:
        return self['m']

    @property
    def v(self) -> Dict[str, Tensor]:
        return self['v']

    def optimize(self, i, param_list: List[DiffedParameter]):
        update_m = [self.m[p.name].write(self.m[p.name] * self.beta1 + p.grad * (F32(1) - self.beta1)) for p in param_list]
        update_v = [self.v[p.name].write(self.v[p.name] * self.beta2 + p.grad.pow(2) * (F32(1) - self.beta2)) for p in param_list]
        a = self.lr * (F32(1) - self.beta2.pow(i)).pow(F32(0.5)) / (F32(1) - self.beta1.pow(i))
        return After(
            when=[update_m, update_v],
            then=[
                p.value.write(p.value - self.m[p.name] / (self.v[p.name].pow(F32(0.5).add(self.eps)))*a)
                for p in param_list
            ])


def train(model, optimizer, inputs, cost, train_while):
    """
    Train a :class:`Differentiable` such as a neural network while the given `train_while` condition is `True`.

    Two named states are provided to `train_while`:
        `i`: the iteration number, a one-indexed `UInt`;
        `output`: a `Tensor`, the last output of the model;
        `loss`: a `Number`, the lats loss of the model's predict.
    """
    
    @closure
    @post_op
    def step(i: UInt, output: Tensor, loss: Tensor):
        loss = cost(output)
        dL = cost(output, dL=True)
        param_list = model.backward(inputs, dL)
        update = optimizer.optimize(i, param_list)
        return After(update, {"i": i + 1, "output": model.forward(inputs).copy(), 'loss': loss})

    output = model.forward(inputs).copy()
    loss = cost(output)
    return While(train_while, step, {"i": 1, "output": output, "loss": loss})
