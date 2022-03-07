"""Basic :class:`Interface` classes."""

from .state import _Base


class Interface(_Base):
    """The base class of a client-defined `Interface`"""


class Equality(Interface):
    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def eq(self, other):
        """Returns `True` if `self` is equal to `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `eq` method")

    def ne(self, other):
        """Returns `True` if `self` is not equal to `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `ne` method")


class Order(Interface):
    @classmethod
    def max(cls, l, r):
        """Return `l`, or `r`, whichever is greater."""

        from .scalar.ref import If
        return cls(If(l >= r, l, r))

    @classmethod
    def min(cls, l, r):
        """Return `l`, or `r`, whichever is lesser."""

        from .scalar.ref import If
        return cls(If(l <= r, l, r))

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.gte(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.lte(other)

    def gt(self, other):
        """Return true if `self` is greater than `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `gt` method")

    def gte(self, other):
        """Return true if `self` is greater than or equal to `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `gte` method")

    def lt(self, other):
        """Return true if `self` is less than `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `lt` method")

    def lte(self, other):
        """Return true if `self` is less than or equal to `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `lte` method")
