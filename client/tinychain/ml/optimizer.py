from abc import abstractmethod
import typing as t

import tinychain as tc
from tinychain.ml import Parameter, DiffedParameter
from tinychain.state import Map
from tinychain.value import F32, UInt

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

        m = Map({p.name: tc.tensor.Dense.zeros(p.ct_shape, F32) for p in param_list})
        v = Map({p.name: tc.tensor.Dense.zeros(p.ct_shape, F32) for p in param_list})

        return cls({"beta1": beta1, "beta2": beta2, "eps": eps, "lr": lr, "m": m, "v": v})

    
        
    def optimize(self, i, param_list: t.List[DiffedParameter], model):
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
        return tc.After(
            when=[update_m, update_v],
            then=update_model)


def train(cxt, model, optimizer, inputs, labels, cost, num_iterations: UInt):


    @tc.closure(model, optimizer, labels, inputs)
    @tc.post_op
    def step(cxt, i: UInt):
        cxt.output = model.forward(inputs).copy()
        cxt.loss = cost(cxt.output, labels)
        cxt.dloss = cost(cxt.output, labels, dl=True)
        param_list = model.backward(inputs, cxt.dloss)
        update = optimizer.optimize(i, param_list, model)
        return tc.After(update, {"i": i + 1, 'loss': cxt.loss})

    @tc.closure(model, optimizer, labels, inputs)
    @tc.post_op
    def cond(i: UInt, loss: tc.tensor.Tensor):
        return i <= num_iterations
    
    return tc.While(cond, step, {"i": 1, "loss": tc.tensor.Dense.ones(labels.shape)})
