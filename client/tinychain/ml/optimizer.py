from abc import abstractmethod

from tinychain.collection.tensor import Dense, Tensor
from tinychain.decorators import closure, post_op
from tinychain.ref import After, While
from tinychain.state import Map
from tinychain.value import F32, Float, UInt


class Optimizer(Map):
    @abstractmethod
    def optimize(self):
        """Update the given `gradient` by computing deltas for the given `loss`."""

    @property
    def lr(self) -> Float:
        return self['lr']


class GradientDescent(Optimizer):
    @classmethod
    def create(cls, lr=F32(0.01)):
        """
        Create a new `GradientDescent` optimizer with the given hyperparameter.

        Args:
            `lr` (float): learning rate
        """

        return cls({"lr": lr})

    def optimize(self, i, param_list):
        return [param.value.write(param.value - (param.grad * self.lr)) for param in param_list]


class Adam(Optimizer):

    @classmethod
    def create(cls, param_list,  beta1=F32(0.9), beta2=F32(0.999), lr=F32(1e-3), eps=F32(1e-8)):
        """
        Create a new `Adam` optimizer with the given hyperparameters.

        Args:
            `param_list` (list): A list of parameters to optimize
            `beta1` (float): The exponential decay rate for the 1st moment estimates;
            `beta2` (float): The exponential decay rate for the 2nd moment estimates;
            `lr` (float): learning rate
            `eps` (float): A small constant for numerical stability.
        """

        m = Map({p.name: Dense.zeros(p.shape, F32) for p in param_list})
        v = Map({p.name: Dense.zeros(p.shape, F32) for p in param_list})

        return cls({"beta1": beta1, "beta2": beta2, "m": m, "v": v, "lr": lr, "eps": eps})

    def optimize(self, i, param_list):
        # TODO: is there a better way to make these accessible to `Map` methods?
        beta1 = self["beta1"]
        beta2 = self["beta2"]
        m = self["m"]
        v = self["v"]
        lr = self["lr"]
        eps = self["eps"]

        update_m = [m[p.name].write(m[p.name] * beta1 + p.grad * (F32(1.0) - beta1)) for p in param_list]
        update_v = [v[p.name].write(v[p.name] * beta2 + p.grad.pow(2) * (F32(1.0) - beta2)) for p in param_list]
        a = lr * F32(F32(1) - beta2.pow(i)).pow(F32(0.5)) / (F32(1) - beta1.pow(i))
        return After(
            [update_m, update_v],
            [
                p.value.write(p.value - m[p.name] / (v[p.name].pow(F32(0.5)).add(eps)) * a)
                for p in param_list
            ])


def train(model, optimizer, inputs, labels, cost, cond):
    @closure(model, optimizer, labels, inputs)
    @post_op
    def step(cxt, i: UInt):
        cxt.output = model.forward(inputs).copy()
        cxt.loss = cost(cxt.output, labels)
        _, param_list = model.backward(inputs, cxt.loss)
        update = optimizer.optimize(i, param_list)
        return After(update, {"i": i + 1, "output": cxt.output})

    return While(cond, step, {"i": 1, "output": model.forward(inputs)})
