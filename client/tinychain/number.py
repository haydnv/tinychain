from .state import OpRef, Value
from .util import uri


# Numeric types

class Number(Value):
    __ref__ = uri(Value) + "/number"

    def __mul__(self, other):
        return self.mul(other)

    def mul(self, other):
        return Number(OpRef.Get(uri(self).append("mul"), other))

