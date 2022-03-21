import logging

from .. import error
from ..app import Dynamic, Model
from ..collection.tensor import Dense, Tensor
from ..decorators import post
from ..math.operator import derivative, Operator
from ..scalar.ref import After
from ..util import form_of


# TODO: support Sparse and Number variable types
class Variable(Dense):
    """A trainable variable in a machine learning model."""

    def __getitem__(self, bounds):
        return Variable(Dense[bounds])

    def cast(self, number_type):
        return Variable(Dense.cast(number_type))

    def expand_dims(self, axis=None):
        return Variable(Dense.expand_dims(self, axis))

    def flip(self, axis):
        return Variable(Dense.flip(axis))

    def update(self, delta):
        """Decrement the value of this `Variable` by `delta`."""

        return self.write(self - delta)

    def reshape(self, shape):
        return Variable(Dense.reshape(self, shape))

    def transpose(self, permutation=None):
        return Variable(Dense.transpose(self, permutation))


class Optimizer(Model):
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
        assert isinstance(form_of(outputs), Operator)

        deltas = backpropagate(loss, form_of(outputs))
        if not deltas:
            logging.warning(f"model {self.ml_model} has no Variables for {self} to train")

        writes = []
        for var, delta in deltas:
            writes.append(var.update((delta * self.lr).sum()))

        return After(writes, loss)


def backpropagate(loss, op):
    assert isinstance(op, Operator)

    deltas = []
    unvisited = [op]

    for op in unvisited:
        assert isinstance(op, Operator)

        if isinstance(op.subject, Variable):
            delta = derivative(loss, op.subject)
            deltas.append((op.subject, delta.copy()))
        elif isinstance(form_of(op.subject), Operator):
            unvisited.append(form_of(op.subject))

        if isinstance(op.args, Variable):
            delta = derivative(loss, op.args)
            deltas.append((op.args, delta.copy()))
        elif isinstance(form_of(op.args), Operator):
            unvisited.append(form_of(op.args))

    return deltas
