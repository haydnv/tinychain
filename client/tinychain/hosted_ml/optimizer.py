import inspect
import logging

from .. import error
from ..app import Dynamic, Model, ModelRef
from ..collection.tensor import Tensor
from ..decorators import post
from ..generic import Map, Tuple
from ..math.operator import derivative, Operator
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
    def __init__(self, ml_model, learning_rate=0.001):
        self.ml_model = ml_model
        self.lr = learning_rate
        Dynamic.__init__(self)

    @post
    def train(self, inputs: Tensor, labels: Tensor) -> Tensor:
        variables = trainable(self.ml_model)
        outputs = self.ml_model.operator(inputs)

        if not isinstance(form_of(outputs), Operator):
            raise ValueError(f"Optimizer can only train a differentiable Operator, not {form_of(outputs)}")

        loss = 0.5 * (outputs - labels)**2  # TODO: support an arbitrary cost function
        gradients = form_of(outputs).gradients(derivative(loss))
        if not gradients:
            logging.warning(f"model {self.ml_model} operator {form_of(outputs)} has no Variables for {self} to train")

        writes = []
        for var_id, delta in gradients.items():
            var = variables[var_id]
            writes.append(var.update(delta * self.lr))

        return After(writes, loss)


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
        for _name, component in inspect.getmembers(model):
            vars.update(trainable(component))

    return vars
