import math

from ..math.interface import Boolean, Complex as _Complex, Numeric, Trigonometric
from ..math.operator import Add, Div, Mul, Sub, Exp, Pow
from ..math.operator import Acos, Acosh, Asin, Asinh, Atan, Atanh, Cos, Cosh, Sin, Sinh, Tan, Tanh
from ..uri import URI

from .ref import deref, form_of, is_literal, same_as
from .value import Value


class Number(Value, Numeric, Trigonometric):
    """A numeric :class:`Value`."""

    __uri__ = URI(Value) + "/number"

    @classmethod
    def trig_rtype(cls):
        return cls

    @property
    def shape(self):
        from ..shape import Shape
        return Shape(tuple())

    def add(self, other):
        """Return the sum of `self` and `other`."""

        if same_as(other, 0):
            return self

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other + self

        return self.__class__(form=Add(self, other))

    def div(self, other):
        """Return the quotient of `self` and `other`."""

        if is_literal((self, other)):
            from ..context import autobox
            return autobox(deref(self) / deref(other))

        if same_as(other, 1):
            return self

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other * (1 / self)

        return self.__class__(form=Div(self, other))

    def exp(self):
        """Calculate `e**self`"""

        if same_as(self, 0):
            return 1
        elif same_as(self, 1):
            return math.e

        return self.__class__(form=Exp(self))

    def modulo(self, other):
        """Return the remainder of `self` divided by `other`."""

        if is_literal((self, other)):
            from ..context import autobox
            return autobox(deref(self) % deref(other))

        if same_as(other, 0):
            raise ValueError(f"divide by zero: {self} / {other}")

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            raise NotImplementedError("Number % Tensor is not supported; " +
                                      f"construct a constant Tensor from {self} instead")

        return self._get("mod", other, self.__class__)

    def mul(self, other):
        """Return the product of `self` and `other`."""

        if is_literal((self, other)):
            from ..context import autobox
            return autobox(deref(self) * deref(other))

        if same_as(other, 1):
            return self

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other * self

        return self.__class__(form=Mul(self, other))

    def pow(self, other):
        """Raise `self` to the power of `other`."""

        if is_literal((self, other)):
            from ..context import autobox
            return autobox(deref(self)**deref(other))

        if same_as(self, 1) or same_as(self, 0) or same_as(other, 1):
            return self

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            raise NotImplementedError("Number**Tensor is not supported; " +
                                      f"construct a constant Tensor from {self} instead")

        return self.__class__(form=Pow(self, other))

    def sub(self, other):
        """Return the difference between `self` and `other`."""

        if is_literal((self, other)):
            from ..context import autobox
            return autobox(deref(self) - deref(other))

        if same_as(other, 0):
            return self

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return -(other - self)

        return self.__class__(form=Sub(self, other))

    def sin(self):
        return self.__class__(form=Sin(self))

    def cos(self):
        return self.__class__(form=Cos(self))

    def asin(self):
        return self.__class__(form=Asin(self))

    def acos(self):
        return self.__class__(form=Acos(self))

    def sinh(self):
        return self.__class__(form=Sinh(self))

    def cosh(self):
        return self.__class__(form=Cosh(self))

    def asinh(self):
        return self.__class__(form=Asinh(self))

    def acosh(self):
        return self.__class__(form=Acosh(self))

    def tan(self):
        return self.__class__(form=Tan(self))

    def tanh(self):
        return self.__class__(form=Tanh(self))

    def atan(self):
        return self.__class__(form=Atan(self))

    def atanh(self):
        return self.__class__(form=Atanh(self))


class Bool(Number, Boolean):
    """A boolean :class:`Value`."""

    __uri__ = URI(Number) + "/bool"

    @classmethod
    def _trig_rtype(cls):
        return F32

    def logical_and(self, other):
        """Boolean AND"""

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.logical_and(self)

        if is_literal((self, other)):
            return Bool(deref(self) and deref(other))

        if same_as(other, 1):
            return self

        return self._get("and", other, Bool)

    def logical_not(self):
        """Boolean NOT"""

        if is_literal(self):
            return Bool(not deref(self))

        if same_as(self, 1):
            return False
        elif same_as(self, 0):
            return True

        return self._get("not", rtype=Bool)

    def logical_or(self, other):
        """Boolean OR"""

        if is_literal((self, other)):
            return Bool(deref(self) or deref(other))

        if same_as(other, 0):
            return self

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.logical_or(self)

        return self._get("or", other, Bool)

    def logical_xor(self, other):
        """Boolean XOR"""

        if is_literal((self, other)):
            return Bool(deref(self) ^ deref(other))

        if same_as(other, 1):
            return self.logical_not()
        elif same_as(other, 0):
            return self

        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.logical_xor(self)

        return self._get("xor", other, Bool)


class Complex(Number, _Complex):
    """A complex number."""

    __uri__ = URI(Number) + "/complex"

    def __init__(self, form, imag=None):
        if imag is not None:
            form = (form, imag)

        if is_literal(form):
            form = deref(form)
            if isinstance(form, complex):
                Number.__init__(self, form)
            else:
                Number.__init__(self, complex(*form))
        else:
            Number.__init__(self, form)

    def __json__(self):
        from ..context import to_json

        form = form_of(self)

        if isinstance(form, complex):
            return {str(URI(type(self))): [[to_json(form.real), to_json(form.imag)]]}
        elif isinstance(form, (list, tuple)):
            assert len(form) == 2
            return {str(URI(type(self))): [to_json(form)]}
        else:
            return Value.__json__(self)

    @property
    def imag(self):
        if is_literal(self):
            return form_of(self).imag
        else:
            return self._get("imag", rtype=Float)

    @property
    def real(self):
        if is_literal(self):
            return form_of(self).real
        else:
            return self._get("real", rtype=Float)

    def conj(self):
        """Return the conjugate of this :class:`Complex` number."""

        return self.__class__(form=(self.real, -self.imag))

    def norm(self):
        """Return the linear norm of this :class:`Complex` number."""

        return self.__class__(form=(self.real**2, self.imag**2))


class C32(Complex):
    """A complex 32-bit floating point number."""

    __uri__ = URI(Complex) + "/32"


class C64(Complex):
    """A complex 64-bit floating point number."""

    __uri__ = URI(Complex) + "/64"


class Float(Number):
    """A floating-point decimal number."""

    __uri__ = URI(Number) + "/float"


class F32(Float):
    """A 32-bit floating point number."""

    __uri__ = URI(Float) + "/32"


class F64(Float):
    """A 64-bit floating point number."""

    __uri__ = URI(Float) + "/64"


class Int(Number):
    """An integer."""

    __uri__ = URI(Number) + "/int"

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

    __uri__ = URI(Int) + "/16"

    @staticmethod
    def size():
        return 16


class I32(Int):
    """A 32-bit integer."""

    __uri__ = URI(Int) + "/32"

    @staticmethod
    def size():
        return 32


class I64(Int):
    """A 64-bit integer."""

    __uri__ = URI(Int) + "/64"

    @classmethod
    def _trig_rtype(cls):
        return F64

    @staticmethod
    def size():
        return 64


class UInt(Number):
    """An unsigned integer."""

    __uri__ = URI(Number) + "/uint"

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

    __uri__ = URI(UInt) + "/8"

    @staticmethod
    def size():
        return 8


class U16(UInt):
    """A 16-bit unsigned integer."""

    __uri__ = URI(UInt) + "/16"

    @staticmethod
    def size():
        return 16


class U32(UInt):
    """A 32-bit unsigned integer."""

    __uri__ = URI(UInt) + "/32"

    @staticmethod
    def size():
        return 32


class U64(UInt):
    """A 64-bit unsigned integer."""

    __uri__ = URI(UInt) + "/64"

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
                return {str(URI(U64)): form}

        return super().__json__()
