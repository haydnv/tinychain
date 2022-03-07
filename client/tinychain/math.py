from .interface import Interface


class Numeric(Interface):
    def __add__(self, other):
        return self.add(other)

    def __abs__(self):
        return self.abs()

    def __radd__(self, other):
        return self.add(other)

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
        return -(self - other)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        return other * (self.pow(-1))

    def abs(self):
        """Absolute value"""

        return self._get("abs", rtype=self.__class__)

    def acos(self):
        """Arccosine"""

        return self._get("acos", rtype=self.__class__)

    def acosh(self):
        """Hyperbolic arccosine"""

        return self._get("acosh", rtype=self.__class__)

    def add(self, other):
        """Addition"""

        return self._post("add", {"r": other}, self.__class__)

    def asin(self):
        """Arcsine"""

        return self._get("asin", rtype=self.__class__)

    def asinh(self):
        """Hyperbolic arcsine"""

        return self._get("asinh", rtype=self.__class__)

    def atan(self):
        """Arctangent"""

        return self._get("atan", rtype=self.__class__)

    def atanh(self):
        """Hyperbolic arctangent"""

        return self._get("atanh", rtype=self.__class__)

    def cos(self):
        """Cosine"""

        return self._get("cos", rtype=self.__class__)

    def cosh(self):
        """Hyperbolic cosine"""

        return self._get("cosh", rtype=self.__class__)

    def div(self, other):
        """Division"""

        return self._post("div", {"r": other}, self.__class__)

    def exp(self):
        """Raise `e` to the power of this `Numeric`."""

        return self._get("exp", rtype=self.__class__)

    def log(self, base=None):
        """
        Calculate the logarithm of this :class:`Numeric` with respect to the given `base`.

        If no base is given, this will return the natural logarithm (base e).
        """

        return self._get("log", base, self.__class__)

    def modulo(self, other):
        """Return the remainder of `self` divided by `other`."""

        return self._get("mod", other, self.__class__)

    def mul(self, other):
        """Multiplication"""

        return self._post("mul", {"r": other}, self.__class__)

    def pow(self, other):
        """Raise this :class:`Numeric` to the given power."""

        return self._post("pow", {"r": other}, self.__class__)

    def round(self, digits=None):
        """
        Round this :class:`Numeric` to the given number of `digits` from the decimal point.

        If `digits` is not specified, this will round to the nearest integer.
        """

        if digits is None:
            return self._get("round", rtype=self.__class__)
        else:
            from .scalar.number import UInt
            places = UInt(10) ** digits
            return (self * places).round() / places

    def sin(self):
        """Sine"""

        return self._get("sin", rtype=self.__class__)

    def sinh(self):
        """Hyperbolic sine"""

        return self._get("sinh", rtype=self.__class__)

    def sub(self, other):
        """Subtraction"""

        return self._post("sub", {"r": other}, self.__class__)

    def tan(self):
        """Tangent"""

        return self._get("tan", rtype=self.__class__)

    def tanh(self):
        """Hyperbolic tangent"""

        return self._get("tanh", rtype=self.__class__)


class Trigonometric(Interface):
    @classmethod
    def trig_rtype(cls):
        raise NotImplementedError(f"trigonometric return type for {cls} is not defined")

    def acos(self):
        """Arccosine"""

        return self._get("acos", rtype=self.trig_rtype())

    def acosh(self):
        """Hyperbolic arccosine"""

        return self._get("acosh", rtype=self.trig_rtype())

    def asin(self):
        """Arcsine"""

        return self._get("asin", rtype=self.trig_rtype())

    def asinh(self):
        """Hyperbolic arcsine"""

        return self._get("asinh", rtype=self.trig_rtype())

    def atan(self):
        """Arctangent"""

        return self._get("atan", rtype=self.trig_rtype())

    def atanh(self):
        """Hyperbolic arctangent"""

        return self._get("atanh", rtype=self.trig_rtype())

    def cos(self):
        """Cosine"""

        return self._get("cos", rtype=self.trig_rtype())

    def cosh(self):
        """Hyperbolic cosine"""

        return self._get("cosh", rtype=self.trig_rtype())

    def sin(self):
        """Sine"""

        return self._get("sin", rtype=self.trig_rtype())

    def sinh(self):
        """Hyperbolic sine"""

        return self._get("sinh", rtype=self.trig_rtype())

    def tan(self):
        """Tangent"""

        return self._get("tan", rtype=self.trig_rtype())

    def tanh(self):
        """Hyperbolic tangent"""

        return self._get("tanh", rtype=self.trig_rtype())
