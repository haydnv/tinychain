import logging

from .. import error
from ..app import Dynamic, Model
from ..collection.tensor import Tensor
from ..decorators import post
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
        outputs = self.ml_model.operator(inputs)
        loss = 0.5 * (outputs - labels)**2  # TODO: support an arbitrary cost function

        if not isinstance(form_of(outputs), Operator):
            raise ValueError(f"Optimizer can only train a differentiable Operator, not {form_of(outputs)}")

        variables = trainable(form_of(outputs))

        gradients = form_of(outputs).gradients(derivative(loss))
        if not gradients:
            logging.warning(f"model {self.ml_model} operator {form_of(outputs)} has no Variables for {self} to train")

        writes = []
        for var_id, delta in gradients.items():
            var = variables[var_id]
            writes.append(var.update(delta * self.lr))

        return After(writes, loss)


def trainable(op):
    assert isinstance(op, Operator)

    vars = {}

    if isinstance(op.subject, Variable):
        vars[hex_id(op.subject)] = op.subject
    elif isinstance(form_of(op.subject), Operator):
        vars.update(trainable(form_of(op.subject)))

    if isinstance(op.args, Variable):
        vars[hex_id(op.args)] = op.args
    elif isinstance(form_of(op.args), Operator):
        vars.update(trainable(form_of(op.args)))

    return vars
