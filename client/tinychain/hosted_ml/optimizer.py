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


class Adam(Optimizer, Dynamic):
    def __init__(self, ml_model, beta1=0.9, beta2=0.999, lr=1e-3, eps=1e-8):
        self.ml_model = ml_model
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps
        self.m = {}
        self.v = {}
        Dynamic.__init__(self)

    @post
    def train(self, inputs: Tensor, labels: Tensor):
        outputs = self.ml_model.operator(inputs)
        loss = (outputs - labels)**2
        dl = 2 * (outputs - labels)
        op_list = [form_of(outputs)]

        assert isinstance(op_list[0], Operator)

        writes = []
        for n, op in enumerate(op_list):
            if isinstance(form_of(op.subject), Operator):
                op_list.append(form_of(op.subject))
            elif isinstance(form_of(op.args), Operator):
                op_list.append(form_of(op.args))

            a = self.lr * (1 - self.beta2.pow(n)).pow(0.5) / (1 - self.beta1.pow(n))

            derivative = op.backward()
            m[-(n+1)].write(m[-(n+1)] * self.beta1 + p.grad * (1.0 - self.beta1))
            v[-(n+1)].write(v[-(n+1)] * self.beta2 + p.grad.pow(2) * (1.0 - self.beta2))

            if isinstance(op.subject, Variable) and isinstance(op.args, Variable):
                raise NotImplementedError(f"partial derivative w/r/t independent variables {op.subject} and {op.args}")
            elif isinstance(op.subject, Variable):
                dl = dl.copy()
                delta = (m[p.name] / (v[p.name].pow(F32(0.5)).add(eps)) * a)
                writes.append(op.subject.update(delta))
            elif isinstance(op.args, Variable):
                dl = dl.copy()
                delta = (m[p.name] / (v[p.name].pow(F32(0.5)).add(eps)) * a)
                writes.append(op.args.update(delta))

        return After(writes, loss)
