import inspect
import logging

from .. import error
from ..app import Dynamic, Model, ModelRef
from ..collection.tensor import Dense, Tensor
from ..decorators import closure, post
from ..generic import Map, Tuple
from ..math.interface import Numeric
from ..math.operator import derivative_of, Operator
from ..scalar.number import F32, F64, Number, UInt
from ..scalar.ref import After, If
from ..state import State, Stream
from ..util import form_of, hex_id

from .variable import Variable
from . import LIB_URI


# this helper class handles the case of both a Number and a Tensor as a gradient
class _Gradient(State, Numeric):
    pass


class Optimizer(Model):
    __uri__ = LIB_URI.append("Optimizer")

    @post
    def train(self, i, inputs, parallel=1):
        return error.NotImplemented(f"{self.__class__.__name__}.train")


class GradientDescent(Optimizer, Dynamic):
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
    def train(self, txn, i: UInt, inputs: Tensor, parallel=UInt(1)) -> Tensor:
        err_parallel = "Optimizer.train requires the batch size {{batch_size}} to be an integer multiple {{parallel}}"
        batch_size = inputs.shape[0]
        txn.chunk_size = UInt(If(
            (parallel > 0).logical_and(batch_size % parallel == 0),
            parallel,
            error.BadRequest(err_parallel, batch_size=batch_size, parallel=parallel)))

        @closure(self, inputs, txn.chunk_size)
        @post
        def sum_grads(cxt, batch: UInt, grads: Map) -> Map:
            cxt.start = batch * txn.chunk_size
            inputs_batch = inputs[cxt.start:(cxt.start + txn.chunk_size)]
            outputs = self.ml_model.eval(inputs_batch)
            operator = form_of(outputs)

            if not isinstance(operator, Operator):
                raise ValueError(f"Optimizer can only train a differentiable Operator, not {form_of(outputs)}")

            loss = self._cost(inputs_batch, outputs)
            cxt.d_loss = derivative_of(loss).copy()
            gradients = operator.gradients(cxt.d_loss)
            assert gradients

            grads = {
                var_id: grads[var_id] + (grad if isinstance(grad, Number) else grad.sum(0))
                for var_id, grad in gradients.items()
            }

            return {"grads": grads}

        trainable_vars = trainable(self.ml_model)
        var_ids = set(trainable_vars.keys())
        gradients = {var_id: 0 for var_id in var_ids}
        gradients = Stream.range((0, batch_size / txn.chunk_size)).fold("batch", {"grads": gradients}, sum_grads)
        gradients = Map(Map(gradients)["grads"])  # TODO: this type information shouldn't get lost

        writes = []
        for var_id, var in trainable_vars.items():
            delta = _Gradient(gradients[var_id])
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
    def train(self, txn, i: UInt, inputs: Tensor, parallel=UInt(1)) -> Tensor:
        err_parallel = "Optimizer.train requires the batch size {{batch_size}} to be an integer multiple {{parallel}}"
        batch_size = inputs.shape[0]
        txn.chunk_size = UInt(If(
            (parallel > 0).logical_and(batch_size % parallel == 0),
            parallel,
            error.BadRequest(err_parallel, batch_size=batch_size, parallel=parallel)))

        @closure(self, inputs, txn.chunk_size)
        @post
        def sum_grads(cxt, batch: UInt, grads: Map) -> Map:
            cxt.start = batch * txn.chunk_size
            inputs_batch = inputs[cxt.start:(cxt.start + txn.chunk_size)]
            outputs = self.ml_model.eval(inputs_batch)
            operator = form_of(outputs)

            if not isinstance(operator, Operator):
                raise ValueError(f"Optimizer can only train a differentiable Operator, not {form_of(outputs)}")

            loss = self._cost(inputs_batch, outputs)
            cxt.d_loss = derivative_of(loss).copy()
            gradients = operator.gradients(cxt.d_loss)
            assert gradients

            grads = {
                var_id: grads[var_id] + (grad if isinstance(grad, Number) else grad.sum(0))
                for var_id, grad in gradients.items()
            }

            return {"grads": grads}

        trainable_vars = trainable(self.ml_model)
        var_ids = set(trainable_vars.keys())
        gradients = {var_id: 0 for var_id in var_ids}
        gradients = Stream.range((0, batch_size / txn.chunk_size)).fold("batch", {"grads": gradients}, sum_grads)
        gradients = Map(Map(gradients)["grads"])  # TODO: this type information shouldn't get lost

        var_names = namespace(self.ml_model, self._model_name)
        var_names = {hex_id(var): name for name, var in var_names.items()}
        gradients = {var_names[var_id]: _Gradient(gradients[var_id]) for var_id in var_names.keys()}

        vars_by_name = {var_names[var_id]: trainable_vars[var_id] for var_id in var_names.keys()}

        update_m = {}
        for name in self.m:
            grad = gradients[name]
            update_m[name] = self.m[name] * self.beta1 * grad * (1. - self.beta1)

        update_v = {}
        for name in self.v:
            grad = gradients[name]
            update_v[name] = self.v[name] * self.beta2 + grad**2 * (1. - self.beta2)

        update_v = {name: self.v[name] * self.beta2 + gradients[name]**2 * (1. - self.beta2) for name in self.v}

        a = self.lr * (1. - self.beta2**i)**0.5 / (1 - self.beta1**i)
        update_model = {name: self.m[name] / (self.v[name]**0.5 + self.eps) * a for name in gradients}

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
