from abc import abstractmethod
from typing import List

from tinychain.collection.tensor import Dense
from tinychain.decorators import closure, post_op
from tinychain.ml import EPS, Parameter
from tinychain.ref import After, While
from tinychain.state import Map
from tinychain.value import F32, UInt


class Optimizer(Map):
    @abstractmethod
    def optimize(self, i, gradient, inputs, loss):
        """Update the given `gradient` by computing deltas for the given `loss`."""


class GradientDescent(Optimizer):
    @classmethod
    def create(cls, lr=F32(0.01)):
        """Create a new `GradientDescent` optimizer with the given `learning_rate`."""

        return cls({"lr": lr})

    def optimize(self, i, param_list: List[Parameter]):
        lr = self["lr"]

        return [param.value.write(param.value - (param.grad * lr)) for param in param_list]


class Adam(Optimizer):
    @classmethod
    def create(cls, beta1=F32(0.9), beta2=F32(0.999), lr=F32(1e-3), eps=F32(1e-8)):
        return cls({'beta1': beta1, 'beta2': beta1, 'lr': lr, 'eps': eps})

    def optimize(self, i, param_list: List[Parameter]):
        self.m = 
        self.v = 
        beta1 = self['beta1']
        beta2 = self['beta2']
        lr = self['lr']
        eps = self['eps']

        self.m = [m.value.write(m.value * beta1 + p.grad * (F32(1) - beta1)) for m, p in zip(self.m, param_list)]

        self.m["w_layer" + str(l)] * beta1 + grads["w_layer" + str(l)]*(F32(1) - beta1)
        self.m["b_layer" + str(l)] * beta1 + grads['b_layer' + str(l)]*(F32(1) - beta1)
        
        self.v["w_layer" + str(l)] * beta2 + grads["w_layer" + str(l)].pow(2) * (F32(1) - beta2)
        self.v["b_layer" + str(l)] * beta2 + grads['b_layer' + str(l)].pow(2) * (F32(1) - beta2)
        
        w_corr = (self.m["w_layer" + str(l)] / (F32(1) - beta1.pow(t)) / ((self.v["w_layer" + str(l)]/(F32(1) - beta2.pow(t))+eps).pow(F32(0.5)))*(lr))
        b_corr = (self.m["b_layer" + str(l)] / (F32(1) - beta1.pow(t)) / ((self.v["b_layer" + str(l)]/(F32(1) - beta2.pow(t)))+eps).pow(F32(0.5))*(lr))
        
        wr = params["b_layer" + str(l)].write(params["b_layer" + str(l)]+b_corr)
        br = params["w_layer" + str(l)].write(params["w_layer" + str(l)]+w_corr)
        return [param.value.write(param.value - (param.grad * lr)) for param in param_list]


def train(model, optimizer, inputs, cost, max_iterations):
    """Train a :class:`Differentiable` such as a neural network in up to `max_iterations` steps."""

    @closure
    @post_op
    def while_cond(i: UInt, loss: Dense):
        fit = (loss > EPS).any()
        return fit.logical_and(i < max_iterations)

    @closure
    @post_op
    def step(i: UInt, loss: Dense):
        loss, param_list = model.backward(inputs, loss)
        writes = optimizer.optimize(i, param_list)
        return After(writes, {"i": i + 1, "loss": loss})

    return While(while_cond, step, {"i": 0, "loss": cost(model.forward(inputs))})
