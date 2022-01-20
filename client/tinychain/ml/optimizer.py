import typing as t
from abc import abstractmethod

from tinychain.collection.tensor import Dense
from tinychain.decorators import closure, post_op
from tinychain.ml import DiffedParameter, Parameter
from tinychain.ref import After, While
from tinychain.state import Map
from tinychain.value import F32, UInt


class Optimizer(Map):
    @abstractmethod
    def optimize(self, param_list: t.List[Parameter]):
        """Update the given `gradient` by computing deltas for the given `loss`."""

    @property
    def lr(self) -> F32:
        return self['lr']


class GradientDescent(Optimizer):
    @classmethod
    def create(cls, lr=F32(0.01)):
        """Create a new `GradientDescent` optimizer with the given hyperparameter.
        Args:
            'lr': learning rate
        """

        return cls({"lr": lr})

    def optimize(self, i, param_list: t.List[Parameter]):
        return [param.value.write(param.value - (param.grad * self.lr)) for param in param_list]


class Adam(Optimizer):

    @classmethod
    def create(cls, param_list: t.List[Parameter],  beta1=F32(0.9), beta2=F32(0.999), lr=F32(1e-3), eps=F32(1e-8)):
        """Create a new `Adam` optimizer with the given hyperparameters.
        Args:
            'lr': learning rate;
            'beta1': The exponential decay rate for the 1st moment estimates;
            'beta2': The exponential decay rate for the 2nd moment estimates;
            'eps': A small constant for numerical stability.
            `param_list`: a `List[Parameter]` of model's parameters for optimizing.
        """

        m = Map({p.name: Dense.zeros(p.value.shape, F32) for p in param_list})
        v = Map({p.name: Dense.zeros(p.value.shape, F32) for p in param_list})

        return cls({"beta1": beta1, "beta2": beta2, "eps": eps, "lr": lr, "m": m, "v": v})

    def optimize(self, i, param_list: t.List[DiffedParameter]):
        beta1 = self["beta1"]
        beta2 = self["beta2"]
        lr = self["lr"]
        eps = self["eps"]
        m = self["m"]
        v = self["v"]
        update_m = [m[p.name].write(m[p.name] * beta1 + p.grad * (F32(1.0) - beta1)) for p in param_list]
        update_v = [v[p.name].write(v[p.name] * beta2 + p.grad.pow(2) * (F32(1.0) - beta2)) for p in param_list]
        a = lr * F32(F32(1) - beta2.pow(i)).pow(F32(0.5)) / (F32(1) - beta1.pow(i))
        update_model = [
                p.value.write(p.value - m[p.name] / (v[p.name].pow(F32(0.5)).add(eps)) * a)
                for p in param_list
                ]
        return After(
            when=[update_m, update_v],
            then=update_model)


def train(model, optimizer, inputs, labels, cost, train_while):

    @closure(model, optimizer, labels, inputs)
    @post_op
    def step(cxt, i: UInt):
        cxt.output = model.forward(inputs).copy()
        cxt.loss = cost(cxt.output, labels)
        cxt.dloss = cost(cxt.output, labels, dl=True)
        _, param_list = model.backward(inputs, cxt.dloss)
        update = optimizer.optimize(i, param_list)

        return After(update, {"i": i + 1, 'loss': cxt.loss, 'output':cxt.output})

    return While(train_while, step, {"i": 1, "loss": F32(1.0), 'output': Dense.ones(labels.shape)})
