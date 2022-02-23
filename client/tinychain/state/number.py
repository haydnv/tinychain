import math

from ..reflect import is_ref
from ..util import form_of, uri

from .base import Interface
from .ref import Operator
from .value import Value


class Numeric(Interface):
    @classmethod
    def _trig_rtype(cls):
        return cls

    def __abs__(self):
        return self.abs()

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __div__(self, other):
        return self.div(other)

    def __mod__(self, other):
        return self.modulo(other)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        return self.pow(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return (-self).add(other)

    def __truediv__(self, other):
        return self.div(other)

    def abs(self):
        """Absolute value"""

        return self._get("abs", rtype=self.__class__)

    def acos(self):
        """Arccosine"""

        return self._get("acos", rtype=self._trig_rtype())

    def acosh(self):
        """Hyperbolic arccosine"""

        return self._get("acosh", rtype=self._trig_rtype())

    def add(self, other):
        """Addition"""

        return self._post("add", {'r': other}, self.__class__)

    def asin(self):
        """Arcsine"""

        return self._get("asin", rtype=self._trig_rtype())

    def asinh(self):
        """Hyperbolic arcsine"""

        return self._get("asinh", rtype=self._trig_rtype())

    def atan(self):
        """Arctangent"""

        return self._get("atan", rtype=self._trig_rtype())

    def atanh(self):
        """Hyperbolic arctangent"""

        return self._get("atanh", rtype=self._trig_rtype())

    def cos(self):
        """Cosine"""

        return self._get("cos", rtype=self._trig_rtype())

    def cosh(self):
        """Hyperbolic cosine"""

        return self._get("cosh", rtype=self._trig_rtype())

    def div(self, other):
        """Division"""

        return self._post("div", {'r': other}, self.__class__)

    def exp(self):
        """Raise `e` to the power of `self`."""

        return self._get("exp", rtype=self.__class__)

    def log(self, base=None):
        """Logarithm with respect to `base`, or `e` if no `base` is given"""

        if base is None or base is math.e:
            return self._get("log", rtype=self.__class__)
        else:
            return self._post("log", {"r": base}, self.__class__)

    def modulo(self, other):
        """The remainder of `self` divided by `other`"""

        return self._post("mod", {'r': other}, self.__class__)

    def mul(self, other):
        """Multiplication"""

        return self._post("mul", {'r': other}, self.__class__)

    def pow(self, other):
        """Exponentiation"""

        return self._post("pow", {'r': other}, self.__class__)

    def round(self, digits=0):
        """Round to `digits` decimal places (`digits` defaults to `0`)"""

        if not digits:
            return self._get("round", rtype=self.__class__)
        else:
            places = UInt(10) ** digits
            return (self * places).round() / places

    def sin(self):
        """Sine"""

        return self._get("sin", rtype=self._trig_rtype())

    def sinh(self):
        """Hyperbolic sine"""

        return self._get("sinh", rtype=self._trig_rtype())

    def sub(self, other):
        """Subtraction"""

        return self._post("sub", {'r': other}, self.__class__)

    def tan(self):
        """Tangent"""

        return self._get("tan", rtype=self._trig_rtype())

    def tanh(self):
        """Hyperbolic tangent"""

        return self._get("tanh", rtype=self._trig_rtype())


class Add(Operator):
    def forward(self):
        return Numeric.add(self.param, self.input)

    def backward(self):
        return differentiate(self.param) + differentiate(self.input)


class Div(Operator):
    def forward(self):
        return Numeric.div(self.param, self.input)

    def backward(self):
        return differentiate(self.param) / differentiate(self.input)


class Mul(Operator):
    def forward(self):
        return Numeric.mul(self.param, self.input)

    def backward(self):
        return (differentiate(self.param) * self.param) + (differentiate(self.input) * self.input)


class Pow(Operator):
    def forward(self):
        return Numeric.pow(self.param, self.input)

    def backward(self):
        return self.input * (self.param**(self.input - 1))


class Sub(Operator):
    def forward(self):
        return Numeric.sub(self.param, self.input)

    def backward(self):
        return differentiate(self.param) - differentiate(self.input)


class Number(Value, Numeric):
    """A numeric :class:`Value`."""

    __uri__ = uri(Value) + "/number"

    def __rdiv__(self, other):
        if _constant(other):
            return Number(other) / self
        else:
            return (1 / self) * other

    def __rtruediv__(self, other):
        if _constant(other):
            return Number(other) / self
        else:
            return (1 / self) * other

    def __rpow__(self, other):
        if _constant(other):
            return Number(other)**self
        else:
            raise TypeError(f"there is no implementation for {other}**{self}")

    def add(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.add(self)

        if _constant(other) == 0:
            return self
        elif _constant(self) == 0:
            return other

        return self.__class__(Add(self, other))

    def div(self, other):
        from ..collection.tensor import Tensor

        if isinstance(other, Tensor):
            return other.mul(1 / self)

        if _constant(other):
            if _constant(other) == 1:
                return self
            elif not _constant(other):
                raise ValueError(f"cannot divide {self} by {other}")

        return self.__class__(Div(self, other))

    def eq(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other == self

        return Value.eq(self, other)

    def gt(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other < self

        return Value.gt(self, other)

    def gte(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other <= self

        return Value.gte(self, other)

    def lt(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other > self

        return Value.lt(self, other)

    def lte(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other >= self

        return Value.lte(self, other)

    def log(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            raise ValueError(f"{self.__class__.__name__}.log({other.__class__.__name__} is not supported; " +
                             "use a constant Tensor instead")

        # TODO: return an Operator
        return Numeric.log(self, other)

    def mul(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.mul(self)

        if _constant(other) == 1:
            return self
        elif _constant(self) == 1:
            return other

        return self.__class__(Mul(self, other))

    def pow(self, other):
        from ..collection.tensor import Tensor

        if isinstance(other, Tensor):
            raise NotImplementedError("Number**Tensor is not supported; " +
                                      f"construct a constant Tensor from {self} instead")

        if _constant(other) == 0:
            return self.__class__(1)
        elif _constant(other) == 1:
            return self
        elif _constant(self) == 1:
            return self

        return self.__class__(Pow(self, other))

    def sub(self, other):
        from ..collection.tensor import Tensor
        if isinstance(other, Tensor):
            return other.add(-self)

        if _constant(other) == 0:
            return self

        return self.__class__(Sub(self, other))


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


def _constant(form):
    form = form_of(form, True)
    if isinstance(form, int) or isinstance(form, float):
        return form


def differentiate(state):
    form = form_of(state, True)

    if isinstance(form, Operator):
        return form.backward()
    elif isinstance(form, int) or isinstance(form, float):
        # `state` is obviously a numeric constant
        return 0
    elif isinstance(state, Numeric) and is_ref(form):
        # assume `state` is a numeric constant
        return 0

    raise ValueError(f"the derivative of {form} is not defined--maybe it should be an Operator?")
