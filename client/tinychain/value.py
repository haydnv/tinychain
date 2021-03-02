""":class:`Value` types such as :class:`Nil`, :class:`Number`, and :class:`String`."""

from .ref import OpRef
from .reflect import Meta
from .state import Scalar
from .util import uri


# Scalar value types

class Value(Scalar, metaclass=Meta):
    """A scalar `Value` which supports equality and collation."""

    __uri__ = uri(Scalar) + "/value"

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def eq(self, other):
        """Returns `true` if `self` is equal to `other`."""

        return _get_op("eq", self, other)

    def ne(self, other):
        """Returns `true` if `self` is not equal to `other`."""

        return _get_op("ne", self, other)


class Nil(Value):
    """A Tinychain `None` Value."""

    __uri__ = uri(Value) + "/none"


class String(Value):
    """A string."""

    __uri__ = uri(Value) + "/string"


# Numeric types

class Number(Value):
    """A numeric :class:`Value`."""

    __uri__ = uri(Value) + "/number"

    def __add__(self, other):
        return self.add(other)

    def __div__(self, other):
        return self.div(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.gte(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.lte(other)

    def __mul__(self, other):
        return self.mul(other)

    def __radd__(self, other):
        return self.add(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __sub__(self, other):
        return self.sub(other)

    def __truediv__(self, other):
        return self.div(other)

    def add(self, other):
        """Return the sum of `self` and `other`."""

        return _get_op("add", self, other)

    def div(self, other):
        """Return the quotient of `self` and `other`."""

        return _get_op("div", self, other)

    def gt(self, other):
        """Return true if `self` is greater than `other`."""

        return _get_op("gt", self, other, Bool)

    def gte(self, other):
        """Return true if `self` is greater than or equal to `other`."""

        return _get_op("gte", self, other, Bool)

    def lt(self, other):
        """Return true if `self` is less than `other`."""

        return _get_op("lt", self, other, Bool)

    def lte(self, other):
        """Return true if `self` is less than or equal to `other`."""

        return _get_op("lte", self, other, Bool)

    def mul(self, other):
        """Return the product of `self` and `other`."""

        return _get_op("mul", self, other)

    def sub(self, other):
        """Return the difference between `self` and `other`."""

        return _get_op("sub", self, other)


def _get_op(name, subject, key, dtype=Number):
    return dtype(OpRef.Get(uri(subject).append(name), key))


class Bool(Number):
    """A boolean :class:`Value`."""

    __uri__ = uri(Number) + "/bool"

