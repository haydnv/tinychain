"""An n-dimensional array of numbers."""

from tinychain.util import to_json, uri
from tinychain.value import F32, Number
from tinychain.collection.collection import Collection
from tinychain.collection.bound import Range

class Tensor(Collection):
    """An n-dimensional array of numbers."""

    __uri__ = uri(Collection) + "/tensor"

    def __getitem__(self, bounds):
        if not isinstance(bounds, tuple):
            bounds = tuple(bounds)

        bounds = [
            Range.from_slice(x) if isinstance(x, slice)
            else x for x in bounds]

        return self._post("", Tensor, bounds=bounds)

    def __add__(self, other):
        return self.add(other)

    def __div__(self, other):
        return self.div(other)

    def __truediv__(self, other):
        return self.div(other)

    def __mul__(self, other):
        return self.mul(other)

    def __sub__(self, other):
        return self.sub(other)

    def abs(self):
        return self._get("abs", rtype=Tensor)

    def add(self, other):
        return self._post("add", Tensor, r=other)

    def all(self):
        """Return `True` if all elements in this `Tensor` are nonzero."""

        return self._get("all", rtype=Tensor)

    def any(self):
        """Return `True` if any element in this `Tensor` are nonzero."""

        return self._get("any", rtype=Tensor)

    def div(self, other):
        """Divide this `Tensor` by another, broadcasting if necessary."""

        return self._post("div", Tensor, r=other)

    def eq(self, other):
        """Return a boolean `Tensor` with element-wise equality values."""

        return self._post("eq", Tensor, r=other)

    def gt(self, other):
        """Return a boolean `Tensor` with element-wise greater-than values."""

        return self._post("gt", Tensor, r=other)

    def gte(self, other):
        """Return a boolean `Tensor` with element-wise greater-or-equal values."""

        return self._post("gte", Tensor, r=other)

    def lt(self, other):
        """Return a boolean `Tensor` with element-wise less-than values."""

        return self._post("lt", Tensor, r=other)

    def lte(self, other):
        """Return a boolean `Tensor` with element-wise less-or-equal values."""

        return self._post("lte", Tensor, r=other)

    def logical_and(self, other):
        """Return a boolean `Tensor` with element-wise logical and values."""

        return self._post("and", Tensor, r=other)

    def logical_not(self):
        """Return a boolean `Tensor` with element-wise logical not values."""

        return self._get("not", rtype=Tensor)

    def logical_or(self, other):
        """Return a boolean `Tensor` with element-wise logical or values."""

        return self._post("or", Tensor, r=other)

    def logical_xor(self, other):
        """Return a boolean `Tensor` with element-wise logical xor values."""

        return self._post("xor", Tensor, r=other)

    def mul(self, other):
        """Multiply this `Tensor` by another, broadcasting if necessary."""

        return self._post("mul", Tensor, r=other)

    def ne(self, other):
        """Return a boolean `Tensor` with element-wise not-equal values."""

        return self._post("ne", Tensor, r=other)

    def product(self, axis=None):
        """
        Calculate the product of this `Tensor` along the given `axis`,
        or the total product if no axis is given.
        """

        rtype = Number if axis is None else Tensor
        return self._get("product", axis, rtype)

    def sub(self, other):
        """Subtract another `Tensor` from this one, broadcasting if necessary."""

        return self._post("sub", Tensor, r=other)

    def sum(self, axis=None):
        """
        Calculate the sum of this `Tensor` along the given `axis`,
        or the total sum if no axis is given.
        """

        rtype = Number if axis is None else Tensor
        return self._get("sum", axis, rtype)

    def write(self, bounds, value):
        """
        Write a `Tensor` or `Number` to the given slice of this one.

        If `bounds` is `None`, this entire `Tensor` will be overwritten.
        """

        return self._put("", bounds, value)
