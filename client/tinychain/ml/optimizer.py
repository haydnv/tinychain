from .. import error
from ..app import Dynamic, Model
from ..collection.tensor import Dense, Tensor
from ..decorators import post
from ..math.operator import constant, derivative_of, is_constant
from ..ml.variable import namespace
from ..scalar.number import Float, F32, F64, UInt
from ..scalar.ref import form_of, After, If

from . import LIB_URI


class Optimizer(Model):
    """An optimizer for a :class:`Differentiable` :class:`Model`"""

    __uri__ = LIB_URI.append("Optimizer")

    @post
    def train(self, i, inputs):
        return error.NotImplemented(f"{self.__class__.__name__}.train")


class GradientDescent(Optimizer, Dynamic):
    """A simple gradient descent optimizer with a configurable learning rate."""

    def __init__(self, ml_model, cost, learning_rate=0.001):
        # compile-time constants
        self._cost = cost
        self._lr = learning_rate

        # run-time state
        self.ml_model = ml_model

        Dynamic.__init__(self)

    @post
    def train(self, cxt, i: UInt, inputs: Tensor) -> Tensor:
        outputs = self.ml_model.eval(inputs)
        d_loss = derivative_of(self._cost(inputs, outputs))
        cxt.d_loss = constant(d_loss.copy() if isinstance(d_loss, Tensor) else d_loss)
        assert is_constant(cxt.d_loss)

        cxt.grads = self.ml_model.gradient(inputs, cxt.d_loss)

        writes = []
        for name, var in namespace(self.ml_model).items():
            grad = cxt.grads[name]
            # TODO: replace `shape.len()` with `ndim`
            delta = Float(If(grad.shape.len() > 0, Tensor(grad).sum(), grad))
            writes.append(var.update(self._lr * delta))

        return writes


class Adam(Optimizer, Dynamic):
    """
    Adam optimizer, an adaptive learning rate optimization algorithm designed to handle sparse gradients and noisy data.

    Based on "Adam: A Method for Stochastic Optimization" by Kingma & Ba, 2014: https://arxiv.org/abs/1412.6980
    """

    def __init__(self, ml_model, cost, beta1=0.9, beta2=0.999, learning_rate=0.001, eps=1e-8):
        # compile-time constants
        self._cost = cost

        # run-time state
        self.ml_model = ml_model
        self.beta1 = F32(beta1)
        self.beta2 = F32(beta2)
        self.lr = F32(learning_rate)
        self.eps = F64(eps)

        self.m = {}
        self.v = {}

        for name, var in namespace(ml_model).items():
            shape = form_of(var.shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError(f"the shape of Variable {name} must be defined at compile time (found {shape})")

            self.m[name] = Dense.constant(shape, 0)
            self.v[name] = Dense.constant(shape, 0)

        Dynamic.__init__(self)

    @post
    def train(self, cxt, i: UInt, inputs: Tensor) -> Tensor:
        assert set(self.m) == set(self.v)

        trainable = namespace(self.ml_model)

        outputs = self.ml_model.eval(inputs)

        d_loss = derivative_of(self._cost(inputs, outputs))
        cxt.d_loss = constant(d_loss.copy() if isinstance(d_loss, Tensor) else d_loss)
        assert is_constant(cxt.d_loss)

        grads = self.ml_model.gradient(inputs, cxt.d_loss)

        cxt.grads = {
            name: Float(If(grads[name].shape.len() > 0, Tensor(grads[name]).sum(), grads[name]))
            for name in self.m
        }

        cxt.update_m = {name: self.m[name] * self.beta1 * cxt.grads[name] * (1. - self.beta1) for name in self.m}
        cxt.update_v = {name: self.v[name] * self.beta2 + cxt.grads[name]**2 * (1. - self.beta2) for name in self.v}

        cxt.a = self.lr * (1. - self.beta2**i)**0.5 / (1 - self.beta1**i)
        cxt.update_model = {name: self.m[name] / (self.v[name]**0.5 + self.eps) * cxt.a for name in self.m}

        updates = After([
            [self.m[name].write(cxt.update_m[name]) for name in self.m],
            [self.v[name].write(cxt.update_v[name]) for name in self.v],
        ], [trainable[name].update(cxt.update_model[name]) for name in self.m])

        return updates
