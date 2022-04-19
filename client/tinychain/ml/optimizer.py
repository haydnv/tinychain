import inspect
import logging

from .. import error
from ..app import Dynamic, Model, ModelRef
from ..collection.tensor import Dense, Tensor
from ..decorators import post
from ..generic import Map, Tuple
from ..math.operator import derivative_of, Operator
from ..scalar.number import F32, F64, UInt
from ..scalar.ref import After, is_op_ref
from ..util import form_of, hex_id

from .variable import Variable
from . import LIB_URI


class Optimizer(Model):
    __uri__ = LIB_URI.append("Optimizer")

    @post
    def train(self, inputs):
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
    def train(self, i: UInt, inputs: Tensor) -> Tensor:
        outputs = self.ml_model.eval(inputs)
        operator = form_of(outputs)

        if not isinstance(operator, Operator):
            raise ValueError(f"Optimizer can only train a differentiable Operator, not {form_of(outputs)}")

        loss = self._cost(inputs, outputs)
        d_loss = derivative_of(loss).copy()
        gradients = operator.gradients(d_loss)

        validate(self.ml_model, self._model_name, operator, gradients)

        variables = trainable(self.ml_model)

        writes = []
        for var_id, delta in gradients.items():
            var = variables[var_id]
            writes.append(var.update(delta * self._lr))

        return After(writes, loss)


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
    def train(self, i: UInt, inputs: Tensor) -> Tensor:
        outputs = self.ml_model.eval(inputs)
        operator = form_of(outputs)

        if not isinstance(operator, Operator):
            raise ValueError(f"Optimizer can only train a differentiable Operator, not {operator}")

        loss = self._cost(inputs, outputs)
        d_loss = derivative_of(loss).copy()
        gradients = operator.gradients(d_loss)

        vars, var_names = validate(self.ml_model, self._model_name, operator, gradients)
        vars = {name: vars[var_id] for var_id, name in var_names.items()}

        gradients = {var_names[var_id]: delta for var_id, delta in gradients.items()}

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
        ], [vars[name].update(delta) for name, delta in update_model.items()])

        return After(updates, loss)


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


def validate(model, name, operator, gradients):
    """Check that the :class:`Variables` of `model` are trainable using the given `operator`."""

    ns = {hex_id(var): name for name, var in namespace(model, name).items()}

    assert isinstance(operator, Operator)
    assert ns

    missing_vars = set(ns.keys())
    vars = {}
    visited = _Queue()
    unvisited = _Queue(operator.subject, operator.args)
    while unvisited:
        node = unvisited.shift()
        logging.debug(f"optimizer traversing Operator graph node {node}")

        if isinstance(form_of(node), Operator):
            assert hex_id(node) not in missing_vars

            node = form_of(node)
            unvisited.push(node.subject)
            unvisited.push(node.args)
            visited.push(node)

        elif isinstance(node, Variable):
            if hex_id(node) in missing_vars:
                logging.debug(f"found Variable {ns[hex_id(node)]}")
            else:
                raise RuntimeError(f"{operator} node {node} references unknown Variable {hex_id(node)} (known: {ns})")

            vars[hex_id(node)] = node
            missing_vars.remove(hex_id(node))

        else:
            logging.debug(f"skipping non-trainable operator graph node {node}")

    if missing_vars:
        missing = set(ns[var_id] for var_id in missing_vars)
        if visited:
            raise RuntimeError(f"{name} operator graph disconnects Variables {missing} at {visited[-1]}")
        else:
            raise RuntimeError(f"{name} operator graph {operator} is not connected to any of its Variables {missing}")

    missing_grads = set(ns[var_id] for var_id in set(ns.keys()) - set(gradients.keys()))
    if missing_grads:
        raise RuntimeError(f"optimizer has gradients for {set(gradients.keys())} but not {missing_grads}")

    extra_grads = set(gradients.keys()) - set(ns.keys())
    if extra_grads:
        raise RuntimeError(f"optimizer found gradients {extra_grads} without corresponding Variables")

    return vars, ns


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
