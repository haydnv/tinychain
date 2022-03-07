from ..math.interface import Numeric, Trigonometric
from ..math.operator import Add, Mul
from ..util import form_of, uri

from .value import Value


# Numeric types

class Number(Value, Numeric, Trigonometric):
    """A numeric :class:`Value`."""

    __uri__ = uri(Value) + "/number"

    @classmethod
    def trig_rtype(cls):
        return cls

    def add(self, other):
        """Return the sum of `self` and `other`."""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other + self

        return self.__class__(Add(self, other))

    def div(self, other):
        """Return the quotient of `self` and `other`."""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other * (1 / self)

        return self._get("div", other, self.__class__)

    def modulo(self, other):
        """Return the remainder of `self` divided by `other`."""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            raise NotImplementedError("Number % Tensor is not supported; " +
                                      f"construct a constant Tensor from {self} instead")

        return self._get("mod", other, self.__class__)

    def mul(self, other):
        """Return the product of `self` and `other`."""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other * self

        return self.__class__(Mul(self, other))

    def pow(self, other):
        """Raise `self` to the power of `other`."""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            raise NotImplementedError("Number**Tensor is not supported; " +
                                      f"construct a constant Tensor from {self} instead")

        return self._get("pow", other, self.__class__)

    def sub(self, other):
        """Return the difference between `self` and `other`."""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return -(other - self)

        return self._get("sub", other, self.__class__)


class Bool(Number):
    """A boolean :class:`Value`."""

    __uri__ = uri(Number) + "/bool"

    @classmethod
    def _trig_rtype(cls):
        return F32

    def logical_and(self, other):
        """Boolean AND"""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.logical_and(self)

        return self._get("and", other, Bool)

    def logical_not(self):
        """Boolean NOT"""

        return self._get("not", rtype=Bool)

    def logical_or(self, other):
        """Boolean OR"""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.logical_or(self)

        return self._get("or", other, Bool)

    def logical_xor(self, other):
        """Boolean XOR"""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.logical_xor(self)

        return self._get("xor", other, Bool)


class Complex(Number):
    """A complex number."""

    __uri__ = uri(Number) + "/complex"

    def abs(self):
        """Return the linear norm of this complex number."""

        return Number.abs(self)

    def norm(self):
        """Return the linear norm of this complex number."""

        return self.abs()


class C32(Complex):
    """A complex 32-bit floating point number."""

    __uri__ = uri(Complex) + "/32"


class C64(Complex):
    """A complex 64-bit floating point number."""

    __uri__ = uri(Complex) + "/64"


class Float(Number):
    """A floating-point decimal number."""

    __uri__ = uri(Number) + "/float"


class F32(Float):
    """A 32-bit floating point number."""

    __uri__ = uri(Float) + "/32"


class F64(Float):
    """A 64-bit floating point number."""

    __uri__ = uri(Float) + "/64"


class Int(Number):
    """An integer."""

    __uri__ = uri(Number) + "/int"

    @classmethod
    def _trig_rtype(cls):
        return F32

    @classmethod
    def max_value(cls):
        """Return the maximum allowed value of this `Int` type."""

        return cls(2 ** (cls.size() - 1) - 1)

    @classmethod
    def min_value(cls):
        """Return the minimum allowed value of this `Int` type."""

        return cls(-2 ** (cls.size() - 1) + 1)

    @staticmethod
    def size():
        return 64


class I16(Int):
    """A 16-bit integer."""

    __uri__ = uri(Int) + "/16"

    @staticmethod
    def size():
        return 16


class I32(Int):
    """A 32-bit integer."""

    __uri__ = uri(Int) + "/32"

    @staticmethod
    def size():
        return 32


class I64(Int):
    """A 64-bit integer."""

    __uri__ = uri(Int) + "/64"

    @classmethod
    def _trig_rtype(cls):
        return F64

    @staticmethod
    def size():
        return 64


class UInt(Number):
    """An unsigned integer."""

    __uri__ = uri(Number) + "/uint"

    @classmethod
    def _trig_rtype(cls):
        return F32

    @classmethod
    def max_value(cls):
        """Return the maximum allowed value of this `Int` type."""

        return cls(2 ** cls.size() - 1)

    @classmethod
    def min_value(cls):
        """Return the minimum allowed value of this `Int` type."""

        return cls(0)

    @staticmethod
    def size():
        return 64


class U8(UInt):
    """An 8-bit unsigned integer (a byte)."""

    __uri__ = uri(UInt) + "/8"

    @staticmethod
    def size():
        return 8


class U16(UInt):
    """A 16-bit unsigned integer."""

    __uri__ = uri(UInt) + "/16"

    @staticmethod
    def size():
        return 16


class U32(UInt):
    """A 32-bit unsigned integer."""

    __uri__ = uri(UInt) + "/32"

    @staticmethod
    def size():
        return 32


class U64(UInt):
    """A 64-bit unsigned integer."""

    __uri__ = uri(UInt) + "/64"

    @classmethod
    def _trig_rtype(cls):
        return F64

    @staticmethod
    def size():
        return 64

    def __json__(self):
        form = form_of(self)
        if isinstance(form, int):
            if form >= 2 ** I32.size():
                return {str(uri(U64)): form}

        return super().__json__()
