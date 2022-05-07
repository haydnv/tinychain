"""Basic :class:`Interface` classes."""

from .base import _Base


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


class Compare(Equality):
    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.gte(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.lte(other)

    def gt(self, other):
        """Return `True` if `self` is greater than `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `gt` method")

    def gte(self, other):
        """Return `True` if `self` is greater than or equal to `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `gte` method")

    def lt(self, other):
        """Return `True` if `self` is less than `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `lt` method")

    def lte(self, other):
        """Return `True` if `self` is less than or equal to `other`."""

        raise NotImplementedError(f"{self.__class__} must implement the `lte` method")


# TODO: generic type support
class Functional(Interface):
    def filter(self, op):
        """Filter the elements of this :class:`Functional` using the given `op`."""

        return self._post("filter", {"op": op}, self.__class__)

    def for_each(self, op):
        """Run the given `op` for each element in this :class:`Functional`, then return the last result.

        This is useful when you need to execute an `op` for its side-effects and not its return value.
        """

        from .state import State
        rtype = op.rtype if hasattr(op, "rtype") else State
        return self._post("for_each", {"op": op}, rtype)

    def fold(self, item_name, initial_state, op):
        """Run the given `op` for each item in this :class:`Functional` along with the previous result.

        `op` must be a POST Op. The item to handle will be passed with the given `item_name` as its name.
        """

        from .state import State
        rtype = type(initial_state) if isinstance(initial_state, State) else State
        return self._post("fold", {"item_name": item_name, "value": initial_state, "op": op}, rtype)

    def map(self, op):
        """Return a new :class:`Functional` with the results of `op` for each element of this :class:`Functional`."""

        return self._post("map", {"op": op}, self.__class__)
