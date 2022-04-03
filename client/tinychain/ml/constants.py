from ..util import URI

LIB_URI = URI("/lib/ml")


def sigmoid(x):
    return 1 / (1 + (-x).exp())
