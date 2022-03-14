import logging

from .. import error
from ..app import Dynamic, Model
from ..collection.tensor import Dense, Tensor
from ..decorators import post
from ..math import Operator
from ..scalar.ref import After
from ..util import form_of


class Variable(Dense):
    def update(self, delta):
        self.write(self - delta)


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

        # TODO: support a custom cost function
        loss = (outputs - labels)**2
        dl = 2 * (outputs - labels)

        op_list = [form_of(outputs)]
        assert isinstance(op_list[0], Operator)

        writes = []
        for op in op_list:
            if isinstance(form_of(op.subject), Operator):
                op_list.append(form_of(op.subject))
            elif isinstance(form_of(op.args), Operator):
                op_list.append(form_of(op.args))

            derivative = op.backward()
            dl = dl / derivative

            if isinstance(op.subject, Variable) and isinstance(op.args, Variable):
                raise NotImplementedError(f"partial derivative w/r/t independent variables {op.subject} and {op.args}")
            elif isinstance(op.subject, Variable):
                dl = dl.copy()
                delta = (dl * self.lr).sum() / inputs.shape[0]
                writes.append(op.subject.update(delta))
            elif isinstance(op.args, Variable):
                dl = dl.copy()
                delta = (dl * self.lr).sum() / inputs.shape[0]
                writes.append(op.args.update(delta))

        if not writes:
            logging.warning(f"model {self.ml_model} has no Variables for {self} to train")

        return After(writes, loss)
