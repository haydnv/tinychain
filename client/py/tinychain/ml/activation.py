from ..collection.tensor import Tensor
from ..math.operator import derivative_of, Unary


def sigmoid(x):
    return 1 / (1 + (-x).exp())


def softmax(x, axis=0):
    p = x.exp()
    return p / p.sum(axis)


class ReLU(Unary):
    def forward(self):
        return (self.subject > 0).cond(self.subject, 0.)

    def backward(self, variable=None):
        d = derivative_of(self.subject, variable)
        return (d > 0).cond(d, 0.)


def relu(x):
    return Tensor(ReLU(x))
