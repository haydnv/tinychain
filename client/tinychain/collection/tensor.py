"""An n-dimensional array of numbers."""

from tinychain import ref
from tinychain.util import uri
from tinychain.value import F32, Number

from . import schema
from .bound import Range
from .collection import Collection


class Tensor(Collection):
    """An n-dimensional array of numbers."""

    __uri__ = uri(Collection) + "/tensor"

    def __getitem__(self, bounds):
        bounds = _handle_bounds(bounds)
        return self._get("", bounds, Tensor)

    def __setitem__(self, bounds, value):
        bounds = _handle_bounds(bounds)
        return self._put("", bounds, value)

    def __add__(self, other):
        return self.add(other)

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

    def __ne__(self, other):
        return self.ne(other)

    def __sub__(self, other):
        return self.sub(other)

    def __truediv__(self, other):
        return self.div(other)

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
        """Calculate the product of this `Tensor` along the given `axis`, or the total product if no axis is given."""

        rtype = Number if axis is None else Tensor
        return self._get("product", axis, rtype)

    def sub(self, other):
        """Subtract another `Tensor` from this one, broadcasting if necessary."""

        return self._post("sub", Tensor, r=other)

    def sum(self, axis=None):
        """Calculate the sum of this `Tensor` along the given `axis`, or the total sum if no axis is given."""

        rtype = Number if axis is None else Tensor
        return self._get("sum", axis, rtype)

    def write(self, *args):
        """
        Write a `Tensor` or `Number` to the given slice of this one.

        If only one argument is provided, it is assumed to be a value to write to this entire `Tensor`.
        If two arguments are provided, the first is assumed to be the bounds of the write and the second the value.
        """

        if len(args) == 1:
            [value] = args
            return self.__setitem__(None, value)
        elif len(args) == 2:
            [bounds, value] = args
            return self.__setitem__(bounds, value)
        else:
            raise TypeError(f"Tensor.write takes exactly one or two arguments, not f{args}")


class Dense(Tensor):
    """An n-dimensional array of numbers stored as sequential blocks."""

    __uri__ = uri(Tensor) + "/dense"

    @classmethod
    def arange(cls, shape, start, stop):
        """
        Return a `DenseTensor` with the given shape containing a range of numbers
        evenly distributed between `start` and `stop`.
        """

        return cls(ref.Get(uri(cls) + "/range", (shape, start, stop)))

    @classmethod
    def constant(cls, shape, value):
        """Return a `DenseTensor` filled with the given `value`."""

        return cls(ref.Get(uri(cls) + "/constant", (shape, value)))

    @classmethod
    def ones(cls, shape, dtype=F32):
        """
        Return a `DenseTensor` filled with ones.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.constant(shape, dtype(1))

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """
        Return a `DenseTensor` filled with zeros.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.constant(shape, dtype(0))


class Sparse(Tensor):
    """An n-dimensional array of numbers stored as a :class:`Table` of coordinates and values."""

    __uri__ = uri(Tensor) + "/sparse"

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """
        Return a sparse tensor with the given shape and data type.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls(schema.Tensor(shape, dtype))

def _handle_bounds(bounds):
    if bounds is None:
        return None

    if hasattr(bounds, "__iter__"):
        bounds = tuple(bounds)
    else:
        bounds = (bounds,)

    return [
        Range.from_slice(x) if isinstance(x, slice)
        else x for x in bounds]
