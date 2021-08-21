""":class:`Value` types such as :class:`Nil`, :class:`Number`, and :class:`String`."""

from tinychain.ref import Ref
from tinychain.state import Scalar
from tinychain.util import form_of, to_json, uri, URI


# Scalar value types

class Value(Scalar):
    """A scalar `Value` which supports equality and collation."""

    __uri__ = uri(Scalar) + "/value"

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def eq(self, other):
        """Returns `true` if `self` is equal to `other`."""

        return self._get("eq", other, Bool)

    def ne(self, other):
        """Returns `true` if `self` is not equal to `other`."""

        return self._get("ne", other, Bool)


class Nil(Value):
    """A `Value` to represent `None`."""

    __uri__ = uri(Value) + "/none"

    def __json__(self):
        form = form_of(self)

        if isinstance(form, Ref) or isinstance(form, URI):
            return to_json(form)
        else:
            return None


class Bytes(Value):
    """A binary `Value`"""

    __uri__ = uri(Value) + "/bytes"


class Id(Value):
    """An identifier"""

    __uri__ = uri(Value) + "/id"

    def __json__(self):
        form = form_of(self)
        if isinstance(form, Ref):
            return to_json(form)
        else:
            return {str(self): []}

    def __str__(self):
        return str(form_of(self))


class String(Value):
    """A string."""

    __uri__ = uri(Value) + "/string"

    def render(self, params=None, **kwargs):
        if kwargs and params is not None:
            raise ValueError("String.render accepts a Map or kwargs, not both:", params, kwargs)

        if params:
            return self._post("render", params, String)
        else:
            return self._post("render", kwargs, String)


# Numeric types

class Number(Value):
    """A numeric :class:`Value`."""

    __uri__ = uri(Value) + "/number"

    def __abs__(self):
        return self.abs()

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __div__(self, other):
        return self.div(other)

    def __eq__(self, other):
        return self.eq(other)

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

    def __rmul__(self, other):
        return self.mul(other)

    def __ne__(self, other):
        return self.ne(other)

    def __pow__(self, other):
        return self.pow(other)

    def __sub__(self, other):
        return self.sub(other)

    def __truediv__(self, other):
        return self.div(other)

    def abs(self):
        """Return this number's absolute value"""

        return self._get("abs", rtype=self.__class__)

    def add(self, other):
        """Return the sum of `self` and `other`."""

        return self._get("add", other, self.__class__)

    def div(self, other):
        """Return the quotient of `self` and `other`."""

        return self._get("div", other, self.__class__)

    def eq(self, other):
        """Return true if `self` is equal to `other`."""

        return self._get("eq", other, Bool)

    def gt(self, other):
        """Return true if `self` is greater than `other`."""

        return self._get("gt", other, Bool)

    def gte(self, other):
        """Return true if `self` is greater than or equal to `other`."""

        return self._get("gte", other, Bool)

    def lt(self, other):
        """Return true if `self` is less than `other`."""

        return self._get("lt", other, Bool)

    def lte(self, other):
        """Return true if `self` is less than or equal to `other`."""

        return self._get("lte", other, Bool)

    def mul(self, other):
        """Return the product of `self` and `other`."""

        return self._get("mul", other, self.__class__)

    def ne(self, other):
        """Return true if `self` is not equal to `other`."""

        return self._get("ne", other, Bool)

    def pow(self, other):
        """Raise `self` to the power of `other`."""

        return self._get("pow", other, self.__class__)

    def sub(self, other):
        """Return the difference between `self` and `other`."""

        return self._get("sub", other, self.__class__)


class Bool(Number):
    """A boolean :class:`Value`."""

    __uri__ = uri(Number) + "/bool"

    def logical_and(self, other):
        """Boolean AND"""

        return self._get("and", other, Bool)

    def logical_not(self):
        """Boolean NOT"""

        return self._get("not", rtype=Bool)

    def logical_or(self, other):
        """Boolean OR"""

        return self._get("or", other, Bool)

    def logical_xor(self, other):
        """Boolean XOR"""

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
    def max(cls):
        return cls(2**(cls.size() - 1) - 1)

    @classmethod
    def min(cls):
        return cls(-2**(cls.size() - 1) + 1)

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

    @staticmethod
    def size():
        return 64


class UInt(Number):
    """An unsigned integer."""

    __uri__ = uri(Number) + "/uint"

    @classmethod
    def max(cls):
        return cls(2**cls.size() - 1)

    @classmethod
    def min(cls):
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

    def __json__(self):
        form = form_of(self)
        if isinstance(form, int):
            if form >= 2**I32.size():
                return {str(uri(U64)): form}

        return super().__json__()

    @staticmethod
    def size():
        return 64
