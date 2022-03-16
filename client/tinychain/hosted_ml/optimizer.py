import inspect
import logging

from .. import error
from ..app import Dynamic, Model
from ..collection.tensor import Dense, Tensor
from ..decorators import post
from ..math import Operator
from ..scalar.ref import After
from ..util import form_of

from .interface import Differentiable


class Variable(Dense):
    def update(self, delta):
        self.write(self - delta)


class Optimizer(Model):
    @post
    def train(self, inputs):
        return error.NotImplemented(f"{self.__class__.__name__}.train")


class GradientDescent(Optimizer, Dynamic):
    def __init__(self, ml_model, learning_rate=0.001):
        self._ns = _namespace(ml_model)

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
                logging.info(f"{self} will update {self._ns[id(op.subject)]}")
            elif isinstance(op.args, Variable):
                dl = dl.copy()
                delta = (dl * self.lr).sum() / inputs.shape[0]
                writes.append(op.args.update(delta))
                logging.info(f"{self} will update {self._ns[id(op.args)]}")

        if not writes:
            logging.warning(f"model {self.ml_model} has no Variables for {self} to train")

        return After(writes, loss)


def _namespace(ml_model):
    variables = {}

    for name, attr in inspect.getmembers(ml_model):
        if name.startswith('_') or (hasattr(attr, "hidden") and attr.hidden):
            continue

        if isinstance(attr, Variable):
            variables[id(attr)] = name
        elif isinstance(attr, Differentiable):
            for var, var_name in _namespace(attr).items():
                variables[var] = f"{name}.{var_name}"
        elif isinstance(attr, dict):
            for item_name in attr:
                for var, var_name in _namespace(attr[item_name]).items():
                    variables[var] = f"{name}.{item_name}.{var_name}"
        elif isinstance(attr, tuple) or isinstance(attr, list):
            for i in range(len(attr)):
                for var, var_name in _namespace(attr[i]).items():
                    variables[var] = f"{name}.{i}.{var_name}"

    return variables
