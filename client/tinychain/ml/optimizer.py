import inspect
import logging
import typing

from .. import error
from ..app import Dynamic, Model, ModelRef
from ..collection.tensor import Dense, Tensor
from ..decorators import post
from ..generic import Map, Tuple
from ..math.interface import Numeric
from ..math.operator import constant, derivative_of, is_constant, simplify
from ..scalar.number import F32, F64, UInt
from ..scalar.ref import form_of, hex_id, After, If
from ..scalar.value import Id
from ..state import State

from .variable import Variable
from . import LIB_URI


# this helper class handles the case of both a Number and a Tensor as a gradient
class _Gradient(State, Numeric):
    pass


_Gradients = typing.Dict[Id, _Gradient]


class Optimizer(Model):
    __uri__ = LIB_URI.append("Optimizer")

    @post
    def train(self, i, inputs):
        return error.NotImplemented(f"{self.__class__.__name__}.train")


class _Optimizer(Optimizer, Dynamic):
    @post
    def gradients(self, cxt, inputs: Tensor) -> _Gradients:
        logging.debug("Optimizer constructing gradient calculations...")

        trainable_vars = trainable(self.ml_model)
        logging.debug("discovered trainable variables...")

        outputs = self.ml_model.eval(inputs)
        logging.debug("constructed model evaluation")

        d_loss = derivative_of(self._cost(inputs, outputs))
        logging.debug("constructed derivative of loss")
        cxt.d_loss = constant(d_loss.copy() if isinstance(d_loss, Tensor) else d_loss)
        assert is_constant(cxt.d_loss)

        grads = {var_id: simplify(derivative_of(outputs, cxt.d_loss, var)) for var_id, var in trainable_vars.items()}
        logging.debug("constructed gradients")

        if not grads:
            raise ValueError(f"model output {outputs} has no gradients")

        return {
            var_id: If(grad.shape[1:] == trainable_vars[var_id].shape, grad.sum(0), grad.sum())
            for var_id, grad in grads.items()}


class GradientDescent(_Optimizer):
    """A simple gradient descent optimizer with a configurable learning rate."""

    def __init__(self, ml_model, cost, learning_rate=0.001):
        # compile-time constants
        self._cost = cost
        self._lr = learning_rate
        self._model_name = ml_model.__class__.__name__

        # run-time state
        self.ml_model = ml_model

        Dynamic.__init__(self)

    @post
    def train(self, txn, i: UInt, inputs: Tensor) -> Tensor:
        grads = self.gradients(inputs)

        writes = []
        for var_id, var in trainable(self.ml_model).items():
            delta = grads[var_id]
            writes.append(var.update(self._lr * delta))

        return writes


class Adam(_Optimizer):
    """
    Adam optimizer, an adaptive learning rate optimization algorithm designed to handle sparse gradients and noisy data.

    Based on "Adam: A Method for Stochastic Optimization" by Kingma & Ba, 2014: https://arxiv.org/abs/1412.6980
    """

    def __init__(self, ml_model, cost, beta1=0.9, beta2=0.999, learning_rate=0.001, eps=1e-8):
        # compile-time constants
        self._cost = cost
        self._model_name = ml_model.__class__.__name__

        # run-time state
        self.ml_model = ml_model
        self.beta1 = F32(beta1)
        self.beta2 = F32(beta2)
        self.lr = F32(learning_rate)
        self.eps = F64(eps)

        self.m = {}
        self.v = {}

        for name, var in namespace(ml_model, self._model_name).items():
            shape = form_of(var.shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError(f"the shape of Variable {name} must be defined at compile time (found {shape})")

            self.m[name] = Dense.constant(shape, 0)
            self.v[name] = Dense.constant(shape, 0)

        Dynamic.__init__(self)

    @post
    def train(self, txn, i: UInt, inputs: Tensor) -> Tensor:
        grads = self.gradients(inputs)

        trainable_vars = trainable(self.ml_model)
        var_names = namespace(self.ml_model, self._model_name)
        var_names = {hex_id(var): name for name, var in var_names.items()}
        grads = {var_names[var_id]: grads[var_id] for var_id in var_names.keys()}

        vars_by_name = {var_names[var_id]: trainable_vars[var_id] for var_id in var_names.keys()}

        update_m = {}
        for name in self.m:
            grad = grads[name]
            update_m[name] = self.m[name] * self.beta1 * grad * (1. - self.beta1)

        update_v = {}
        for name in self.v:
            grad = grads[name]
            update_v[name] = self.v[name] * self.beta2 + grad**2 * (1. - self.beta2)

        update_v = {name: self.v[name] * self.beta2 + grads[name]**2 * (1. - self.beta2) for name in self.v}

        a = self.lr * (1. - self.beta2**i)**0.5 / (1 - self.beta1**i)
        update_model = {name: self.m[name] / (self.v[name]**0.5 + self.eps) * a for name in grads}

        updates = After([
            [self.m[name].write(new_value) for name, new_value in update_m.items()],
            [self.v[name].write(new_value) for name, new_value in update_v.items()],
        ], [vars_by_name[name].update(delta) for name, delta in update_model.items()])

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


def namespace(model, prefix):
    """Traverse the attributes of the given `model` to create a namespace for its trainable :class:`Variable` s."""

    if isinstance(model, Variable):
        return {prefix: model}
    elif isinstance(model, ModelRef):
        return namespace(model.instance, prefix)

    if isinstance(model, (Map, Tuple)):
        model = form_of(model)

    ns = {}

    if isinstance(model, (list, tuple)):
        for i, component in enumerate(model):
            ns.update(namespace(component, f"{prefix}.{i}"))
    elif isinstance(model, dict):
        for name, component in model.items():
            ns.update(namespace(component, f"{prefix}.{name}"))
    elif isinstance(model, Model):
        for name, component in inspect.getmembers(model):
            if name.startswith("__"):
                continue

            ns.update(namespace(component, f"{prefix}.{name}"))
    else:
        logging.debug(f"ignoring non-trainable model attribute {model}")

    return ns


def trainable(model):
    """Traverse the attributes of the given `model` to discover its trainable :class:`Variable` s."""

    if isinstance(model, Variable):
        return {hex_id(model): model}
    elif isinstance(model, ModelRef):
        return trainable(model.instance)

    if isinstance(model, (Map, Tuple)):
        model = form_of(model)

    vars = {}

    if isinstance(model, (list, tuple)):
        for component in model:
            vars.update(trainable(component))
    elif isinstance(model, dict):
        for component in model.values():
            vars.update(trainable(component))
    elif isinstance(model, Model):
        for name, component in inspect.getmembers(model):
            if name.startswith("__"):
                continue

            vars.update(trainable(component))
    else:
        logging.debug(f"ignoring non-trainable model attribute {model}")

    return vars
