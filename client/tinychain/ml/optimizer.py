import inspect
import logging

from .. import error
from ..app import Dynamic, Model, ModelRef
from ..collection.tensor import Dense, Tensor
from ..decorators import post
from ..generic import Map, Tuple
from ..math.operator import derivative_of, Operator
from ..scalar.number import F32, F64, UInt
from ..scalar.ref import After
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

    def __init__(self, ml_model, learning_rate=0.001):
        self.ml_model = ml_model
        self.lr = F32(learning_rate)
        Dynamic.__init__(self)

    @post
    def train(self, i: UInt, inputs: Tensor, labels: Tensor) -> Tensor:
        outputs = self.ml_model.eval(inputs)
        if not isinstance(form_of(outputs), Operator):
            raise ValueError(f"Optimizer can only train a differentiable Operator, not {form_of(outputs)}")

        loss = (outputs - labels)**2  # TODO: support an arbitrary cost function
        gradients = form_of(outputs).gradients(derivative_of(loss))

        if not gradients:
            logging.warning(f"model {self.ml_model} operator {form_of(outputs)} has no Variables for {self} to train")

        variables = trainable(self.ml_model)

        writes = []
        for var_id, delta in gradients.items():
            var = variables[var_id]
            writes.append(var.update(delta * self.lr))

        return After(writes, loss)


class Adam(Optimizer, Dynamic):
    """
    Adam optimizer, an adaptive learning rate optimization algorithm designed to handle sparse gradients and noisy data.

    Based on "Adam: A Method for Stochastic Optimization" by Kingma & Ba, 2014: https://arxiv.org/abs/1412.6980
    """

    def __init__(self, ml_model, beta1=0.9, beta2=0.999, learning_rate=0.001, eps=1e-8):
        self._ns = namespace(ml_model, ml_model.__class__.__name__)
        if not self._ns:
            raise ValueError(f"{ml_model} has no Variables to train")

        self.ml_model = ml_model
        self.beta1 = F32(beta1)
        self.beta2 = F32(beta2)
        self.lr = F32(learning_rate)
        self.eps = F64(eps)

        self.m = Map({name: Dense.zeros(var.shape) for name, var in self._ns.items()})
        self.v = Map({name: Dense.zeros(var.shape) for name, var in self._ns.items()})

        Dynamic.__init__(self)

    @post
    def train(self, cxt, i: UInt, inputs: Tensor, labels: Tensor) -> Tensor:
        var_names = {hex_id(var): name for name, var in self._ns.items()}

        outputs = self.ml_model.eval(inputs)

        if not isinstance(form_of(outputs), Operator):
            raise ValueError(f"Optimizer can only train a differentiable Operator, not {form_of(outputs)}")

        loss = (outputs - labels)**2  # TODO: support an arbitrary cost function
        gradients = form_of(outputs).gradients(derivative_of(loss))
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

        cxt.a = self.lr * (1. - self.beta2**i)**0.5 / (1 - self.beta1**i)
        update_model = {name: self.m[name] / (self.v[name]**0.5 + self.eps) * cxt.a for name in gradients}

        updates = After([
            [self.m[name].write(new_value) for name, new_value in update_m.items()],
            [self.v[name].write(new_value) for name, new_value in update_v.items()],
        ], [self._ns[name].update(delta) for name, delta in update_model.items()])

        return After(updates, loss)


def namespace(model, prefix):
    if isinstance(model, Variable):
        return {prefix: model}

    if isinstance(model, Map) or isinstance(model, Tuple):
        model = form_of(model)

    ns = {}

    if isinstance(model, list) or isinstance(model, tuple):
        for i, component in enumerate(model):
            ns.update(namespace(component, f"{prefix}.{i}"))
    elif isinstance(model, dict):
        for name, component in model.items():
            ns.update(namespace(component, f"{prefix}.{name}"))
    elif isinstance(model, Model) or isinstance(model, ModelRef):
        # TODO: a ModelRef should implement the same interfaces as its Model
        for name, component in inspect.getmembers(model):
            if name.startswith("__"):
                continue

            ns.update(namespace(component, f"{prefix}.{name}"))
    else:
        logging.debug(f"ignoring non-trainable model component {model}")

    return ns


def trainable(model):
    if isinstance(model, Variable):
        return {hex_id(model): model}

    if isinstance(model, Map) or isinstance(model, Tuple):
        model = form_of(model)

    vars = {}

    if isinstance(model, list) or isinstance(model, tuple):
        for component in model:
            vars.update(trainable(component))
    elif isinstance(model, dict):
        for component in model.values():
            vars.update(trainable(component))
    elif isinstance(model, Model) or isinstance(model, ModelRef):
        # TODO: a ModelRef should implement the same interfaces as its Model
        for name, component in inspect.getmembers(model):
            if name.startswith("__"):
                continue

            vars.update(trainable(component))
    else:
        logging.debug(f"ignoring non-trainable model component {model}")

    return vars
