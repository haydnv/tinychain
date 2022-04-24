from ..collection.tensor import where, Tensor
from ..math.operator import derivative_of, Unary


def sigmoid(x):
    return 1 / (1 + (-x).exp())


def softmax(x, axis=0):
    p = x.exp()
    return p / p.sum(axis)


# TODO: should there be a general "Where" operator?
class ReLU(Unary):
    def forward(self):
        return where(self.subject > 0, self.subject, 0.)

    def backward(self, variable=None):
        d = derivative_of(self.subject, variable)
        return where(d > 0, d, 0.)


def relu(x):
    return Tensor(ReLU(x))
