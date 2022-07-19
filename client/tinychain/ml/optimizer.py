import logging

from .. import error
from ..app import Dynamic, Model
from ..collection.tensor import Dense, Tensor
from ..decorators import post
from ..generic import Map, Tuple
from ..math.operator import constant, derivative_of, is_constant
from ..ml.interface import Gradients
from ..ml.variable import namespace
from ..scalar.number import Float, F32, F64, UInt
from ..scalar.op import Post
from ..scalar.ref import form_of, After, If

from . import LIB_URI


class Optimizer(Model):
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
        logging.debug("constructed model evaluation")

        d_loss = derivative_of(self._cost(inputs, outputs))
        logging.debug("constructed derivative of loss")
        cxt.d_loss = constant(d_loss.copy() if isinstance(d_loss, Tensor) else d_loss)
        assert is_constant(cxt.d_loss)

        # TODO: these type expectations & keyword arguments should not be necessary
        grads = self.ml_model.gradient(inputs=inputs, loss=cxt.d_loss)
        grads = Tuple.expect((Map.expect(Gradients), Post))(grads)
        cxt.grads, _grad_fn = grads

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
        logging.debug("constructed model evaluation")

        d_loss = derivative_of(self._cost(inputs, outputs))
        logging.debug("constructed derivative of loss")
        cxt.d_loss = constant(d_loss.copy() if isinstance(d_loss, Tensor) else d_loss)
        assert is_constant(cxt.d_loss)

        # TODO: this type expectation & keyword arguments should not be necessary
        cxt.grads = Map.expect(Gradients)(self.ml_model.gradient(inputs=inputs, loss=cxt.d_loss))

        update_m = {}
        for name in self.m:
            grad = cxt.grads[name]
            # TODO: replace `shape.len()` with `ndim`
            grad = Float(If(grad.shape.len() > 0, Tensor(grad).sum(), grad))
            update_m[name] = self.m[name] * self.beta1 * grad * (1. - self.beta1)

        cxt.update_m = update_m

        update_v = {}
        for name in self.v:
            grad = cxt.grads[name]
            update_v[name] = self.v[name] * self.beta2 + grad**2 * (1. - self.beta2)

        cxt.update_v = {name: self.v[name] * self.beta2 + cxt.grads[name]**2 * (1. - self.beta2) for name in self.v}

        cxt.a = self.lr * (1. - self.beta2**i)**0.5 / (1 - self.beta1**i)
        cxt.update_model = {name: self.m[name] / (self.v[name]**0.5 + self.eps) * cxt.a for name in self.m}

        updates = After([
            [self.m[name].write(cxt.update_m[name]) for name in self.m],
            [self.v[name].write(cxt.update_v[name]) for name in self.v],
        ], [trainable[name].update(cxt.update_model[name]) for name in self.m])

        return updates


class _Queue(object):
    def __init__(self, *nodes):
        self._queue = []

        for node in nodes:
            self.push(node)

    def __bool__(self):
        return bool(self._queue)

    def __getitem__(self, key):
        return self._queue[key]

    def __repr__(self):
        return str(self._queue)

    def push(self, node):
        if node is None:
            return
        elif isinstance(node, (Map, Tuple)):
            return self.push(form_of(node))

        if isinstance(node, (list, tuple)):
            for item in node:
                self.push(item)
        elif isinstance(node, dict):
            for item in node.values():
                self.push(item)
        else:
            self._queue.append(node)

    def shift(self):
        return self._queue.pop(0)
