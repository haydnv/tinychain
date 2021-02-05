from .state import OpRef, Scalar
from .util import uri

# Scalar value types

class Value(Scalar):
    __ref__ = uri(Scalar) + "/value"

    def __eq__(self, other):
        return self.eq(other)

    def eq(self, other):
        return _get_op("eq", self, other, Bool)


class Nil(Value):
    __ref__ = uri(Value) + "/none"


# Numeric types

class Number(Value):
    __ref__ = uri(Value) + "/number"

    def __init__(self, ref):
        Value.__init__(self, ref)

    def __add__(self, other):
        return self.add(other)

    def __gt__(self, other):
        return self.gt(other)

    def __gte__(self, other):
        return self.gte(other)

    def __lt__(self, other):
        return self.lt(other)

    def __lte__(self, other):
        return self.lte(other)

    def __mul__(self, other):
        return self.mul(other)

    def add(self, other):
        return _get_op("add", self, other)

    def gt(self, other):
        return _get_op("gt", self, other, Bool)

    def gte(self, other):
        return _get_op("gte", self, other, Bool)

    def lt(self, other):
        return _get_op("lt", self, other, Bool)

    def lte(self, other):
        return _get_op("lte", self, other, Bool)

    def mul(self, other):
        return _get_op("mul", self, other)


def _get_op(name, subject, key, dtype=Number):
    return dtype(OpRef.Get(uri(subject).append(name), key))


class Bool(Number):
    __ref__ = uri(Number) + "/bool"

