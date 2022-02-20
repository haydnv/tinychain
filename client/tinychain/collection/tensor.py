"""An n-dimensional array of numbers."""

import typing

from ..decorators import post_op
from ..reflect import is_ref
from ..state.generic import Map, Tuple
from ..state.number import Add, Div, Mul, Sub, Pow, Numeric, Number, Bool, F32, F64, UInt
from ..state import ref, Class, State, Stream
from ..util import form_of, uri, URI

from .bound import Range
from .base import Collection


class Tensor(Collection):
    """An n-dimensional array of numbers."""

    __uri__ = uri(Collection) + "/tensor"
    __spec__ = (typing.Tuple[U64, ...], Number)

    @classmethod
    def expect(cls, shape, dtype):
        spec = (shape, dtype)

        if not isinstance(shape, Tuple):
            shape = Tuple(shape)

        class _Tensor(cls):
            __spec__ = spec

            @classmethod
            def create(cls):
                return cls(ref.Get(cls, (shape, dtype)))

            @property
            def dtype(self):
                return dtype

            @property
            def schema(self):
                return shape, dtype

            @property
            def shape(self):
                return shape

        return _Tensor

    @classmethod
    def create(cls, shape, dtype=F32):
        """
        Create a new, empty :class:`Tensor`.

        Call this method to initialize a persistent :class:`Tensor` in a :class:`Chain`.
        """

        return cls.expect(shape, dtype).create()

    @classmethod
    def load(cls, shape, dtype, data):
        """
        Load a `Tensor` from an existing data set.

        Example:
            .. highlight:: python
            .. code-block:: python

                coords = [[0, 0, 1], [0, 1, 0]]
                values = [1, 2]
                sparse = tc.tensor.Sparse.load([2, 3, 4], tc.I32, zip(coords, values))
                dense = tc.tensor.Dense.load([2, 3, 4], tc.i32, values)
        """

        return super().load((shape, dtype), data)

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

    def __ne__(self, other):
        return self.ne(other)

    def __getitem__(self, bounds):
        parent = self
        bounds = _handle_bounds(bounds)

        class Slice(self.__class__):
            def write(self, value):
                return parent._put("", bounds, value)

        return self._get("", bounds, Slice)

    def __setitem__(self, bounds, value):
        raise NotImplementedError("use Tensor.write instead")

    def __rdiv__(self, other):
        return (self**-1) * other

    def __rtruediv__(self, other):
        return (self**-1) * other

    def add(self, other):
        """Return the element-wise sum of this `Tensor` and another `Tensor` or `Number`."""

        return Tensor(Add(self, other))

    def all(self):
        """Return `True` if all elements in this `Tensor` are nonzero."""

        return self._get("all", rtype=Bool)

    def any(self):
        """Return `True` if any element in this `Tensor` are nonzero."""

        return self._get("any", rtype=Bool)

    def argmax(self, axis=None):
        """Return the indices of the maximum values along the given `axis` of this `Tensor`.

        If no `axis` is given, the total offset will be returned.
        """

        return self._get("argmax", axis, self.__class__)

    def cast(self, number_type):
        """Cast the data type of `Tensor` into the given `number_type`."""

        return self._get("cast", number_type, self.__class__)

    def copy(self):
        """Return a copy of this `Tensor`"""

        return self.__class__(ref.Post(uri(Tensor) + "/copy_from", {"tensor": self}))

    def div(self, other):
        """Divide this `Tensor` by another `Tensor` or `Number`, broadcasting if necessary."""

        return Tensor(Div(self, other))

    @property
    def dtype(self):
        """Return the data type of this `Tensor`."""

        return self._get("dtype", rtype=Class)

    def flip(self, axis):
        """Flip the elements in this `Tensor` along the specified `axis`."""

        return self._get("flip", axis, Tensor)

    def eq(self, other):
        """Return a boolean `Tensor` with element-wise equality values."""

        return self._post("eq", {"r": other}, Tensor)

    def exp(self):
        """Raise `e` to the power of this `Tensor`."""

        return self._get("exp", rtype=self.__class__)

    def expand_dims(self, axis=None):
        """Return a view of this `Tensor` with an extra dimension of size 1 at the given axis."""

        return self._get("expand_dims", axis, self.__class__)

    def log(self, base=None):
        """Logarithm with respect to `base`, or `e` if no `base` is given"""

        return self._get("log", base, self.__class__)

    def lt(self, other):
        """Return a boolean `Tensor` with element-wise less-than values."""
        return self._post("lt", {"r": other}, self.__class__)

    def lte(self, other):
        """Return a boolean `Tensor` with element-wise less-or-equal values."""

        return self._post("lte", {"r": other}, Tensor)

    def gt(self, other):
        """Return a boolean `Tensor` with element-wise greater-than values."""

        return self._post("gt", {"r": other}, self.__class__)

    def gte(self, other):
        """Return a boolean `Tensor` with element-wise greater-or-equal values."""

        return self._post("gte", {"r": other}, Tensor)

    def logical_and(self, other):
        """Return a boolean `Tensor` with element-wise logical and values."""

        return self._post("and", {"r": other}, self.__class__)

    def logical_not(self):
        """Return a boolean `Tensor` with element-wise logical not values."""

        return self._get("not", rtype=Tensor)

    def logical_or(self, other):
        """Return a boolean `Tensor` with element-wise logical or values."""

        return self._post("or", {"r": other}, Tensor)

    def logical_xor(self, other):
        """Return a boolean `Tensor` with element-wise logical xor values."""

        return self._post("xor", {"r": other}, Tensor)

    @property
    def ndim(self):
        """Return the number of dimensions of this `Tensor`."""

        return self._get("ndim", rtype=UInt)

    def mean(self, axis=None):
        """
        Return the average of this `Tensor` along the given `axis`,
        or the average of the entire `Tensor` if no axis is given.
        """

        if axis is None:
            return self.sum() / self.size
        else:
            return self.sum(axis) / self.shape[axis]

    def mul(self, other):
        """Multiply this `Tensor` by another `Tensor` or `Number`, broadcasting if necessary."""

        return self.__class__(Mul(self, other))

    def ne(self, other):
        """Return a boolean `Tensor` with element-wise not-equal values."""

        return self._post("ne", {"r": other}, self.__class__)

    def pow(self, other):
        """Raise this `Tensor` to the given power."""

        return self.__class__(Pow(self, other))

    def product(self, axis=None):
        """Calculate the product of this `Tensor` along the given `axis`, or the total product if no axis is given."""

        rtype = Number if axis is None else self.__class__
        return self._get("product", axis, rtype)

    def reshape(self, shape):
        """Return a view of this `Tensor` with the given `shape`."""

        return self._get("reshape", shape, self.__class__)

    def round(self):
        """Round this `Tensor` to the nearest integer, element-wise."""

        return self._get("round", rtype=self.__class__)

    @property
    def shape(self):
        """Return the shape of this `Tensor`."""

        return self._get("shape", rtype=Tuple.expect(typing.Tuple[U64, ...]))

    @property
    def size(self):
        """Return the size of this `Tensor` (the product of its `shape`)."""

        return self._get("size", rtype=UInt)

    def split(self, num_or_size_splits, axis=0):
        """
        Split this `Tensor` into multiple slices along the given `axis`.

        If `num_or_size_splits` is a `Number`, the `tensor` will be sliced along `axis` `num_or_size_splits` times;
        if `self.shape[axis] % num_or_size_splits != 0` then a `BadRequest` error will be raised.

        If `num_or_size_splits` is a `Tuple` with length `n` then the `tensor` will be split into `n` new `Tensor` s
        each with `shape[axis] == num_or_size_splits[axis]`; if the sum of `num_or_size_splits` is not equal to
        `self.shape[axis]` then a `BadRequest` error will be raised.
        """

        return self._get("split", (num_or_size_splits, axis), Tuple)

    def std(self, axis=None):
        """
        Return the standard deviation of this `Tensor` along the given `axis`,
        or the standard deviation of the entire `Tensor` if no axis is given.
        """

        if axis is None:
            average = self.mean()
            size = self.size
            return (((self - average)**2).sum() / size)**0.5
        else:
            raise NotImplementedError("Tensor.std with axis")

    def sub(self, other):
        """Subtract another `Tensor` or `Number` from this one, broadcasting if necessary."""

        return Tensor(Sub(self, other))

    def sum(self, axis=None):
        """Calculate the sum of this `Tensor` along the given `axis`, or the total sum if no axis is given."""

        rtype = Number if axis is None else self.__class__
        return self._get("sum", axis, rtype)

    def transpose(self, permutation=None):
        """
        Return a view of this `Tensor` with its axes transposed according to the given permutation.

        If no permutation is given, the axes will be inverted (e.g. `(0, 1, 2)` inverts to `(2, 1, 0)`).
        """

        return self._get("transpose", permutation, self.__class__)

    def write(self, value):
        """Overwrite this `Tensor` with the given `Tensor` or `Number`, broadcasting if needed."""

        return self._put("", None, value)


class Dense(Tensor):
    """
    An n-dimensional array of numbers stored as sequential blocks.

    **IMPORTANT**: for efficiency reasons, serialization of a `Dense` tensor will stop if a non-numeric value
    (NaN or +/- infinity) is encountered. If you receive a `Dense` tensor without enough elements for its shape,
    you can safely treat this response as a divide-by-zero error.
    """

    __uri__ = uri(Tensor) + "/dense"

    @classmethod
    def arange(cls, shape, start, stop):
        """
        Return a `Dense` tensor with the given shape containing a range of numbers
        evenly distributed between `start` and `stop`.
        """

        dtype = type(start) if isinstance(start, Number) else Number
        return cls.expect(shape, dtype)(ref.Get(uri(cls) + "/range", (shape, start, stop)))

    @classmethod
    def concatenate(cls, tensors, axis=None):
        """Create a new `Dense` tensor by concatenating the given `tensors` along the given `axis`."""

        params = {"tensors": tensors}
        if axis:
            params["axis"] = axis

        return cls(ref.Post(uri(cls) + "/concatenate", params))

    @classmethod
    def constant(cls, shape, value):
        """Return a `Dense` tensor filled with the given `value`."""

        dtype = type(value) if isinstance(value, Number) else Number
        return cls.expect(shape, dtype)(ref.Get(uri(cls) + "/constant", (shape, value)))

    @classmethod
    def load(cls, shape, dtype, data):
        """
        Load a `Dense` tensor from an existing data set.

        Example: `tc.tensor.Dense.load([2, 2], tc.i32, [1, 2, 3, 4])`
        """

        if is_ref(shape) or is_ref(dtype):
            raise ValueError(f"cannot load schema ({shape}, {dtype}) (consider calling `copy_from` instead)")

        if is_ref(data):
            raise ValueError(f"cannot load data {data} (consider calling `copy_from` instead)")

        class Load(cls.expect(shape, dtype)):
            def __ref__(self, name):
                return cls(URI(name))

        return Load(ref.Put(cls, (shape, dtype), data))

    @classmethod
    def ones(cls, shape, dtype=F32):
        """
        Return a `Dense` tensor filled with ones.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.expect(shape, dtype).constant(shape, Number(1).cast(dtype))

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """
        Return a `Dense` tensor filled with zeros.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.expect(shape, dtype).constant(shape, Number(0).cast(dtype))

    @classmethod
    def random_normal(cls, shape, mean=0.0, std=1.0):
        """Return a `Dense` tensor filled with a random normal distribution of `F64` s."""

        args = {"shape": shape, "mean": mean, "std": std}
        return cls.expect(shape, F64)(ref.Post(uri(cls) + "/random/normal", args))

    @classmethod
    def random_uniform(cls, shape, minval=0, maxval=1):
        """Return a `Dense` tensor filled with a uniform random distribution of `F64` s."""

        if minval == maxval:
            return cls.constant(shape, minval)

        assert maxval > minval

        random = cls(ref.Get(uri(cls) + "/random/uniform", shape))
        if (minval, maxval) == (0, 1):
            return random
        else:
            range = maxval - minval
            return (random * range) + minval

    @classmethod
    def truncated_normal(cls, shape, mean=0.0, std=1.0, minval=None, maxval=None):
        """
        Return a `Dense` tensor filled with a random normal distribution of `F64` s.

        Any value `x` outside the range `minval <= x <= maxval` will be replaced by a value drawn from a new
        random normal distribution.

        `minval` and `maxval` default to two standard deviations.
        """

        if not std:
            return cls.constant(shape, mean)

        minval = std * -2 if minval is None else minval
        maxval = std * 2 if maxval is None else maxval

        @post_op
        def cond(cxt, tensor: Dense) -> Bool:
            cxt.too_small = (tensor < minval).any()
            cxt.too_big = (tensor > maxval).any()
            return cxt.too_small.logical_or(cxt.too_big)

        @post_op
        def step(cxt, tensor: Dense) -> Map:
            cxt.new_dist = Dense.random_normal(shape, mean, std)
            cxt.new_tensor = where((tensor >= minval).logical_and(tensor <= maxval), tensor, cxt.new_dist)
            return Map(tensor=cxt.new_tensor.copy())

        truncated = Map(ref.While(cond, step, Map(tensor=Dense.random_normal(shape, mean, std))))["tensor"]
        return cls(truncated)

    def argsort(self):
        """Return the coordinates needed to sort this `Tensor`."""

        return self._get("argsort", rtype=self.__class__)

    def elements(self, bounds):
        """Return a :class:`Stream` of the :class:`Number` elements of this `Dense` tensor."""

        bounds = _handle_bounds(bounds)
        return self._get("elements", bounds, Stream)

    def as_sparse(self):
        """Return a :class:`Sparse` view of this `Dense` tensor."""

        return self._get("sparse", rtype=Sparse)


class Sparse(Tensor):
    """
    An n-dimensional array of numbers stored as a :class:`Table` of coordinates and values.

    **IMPORTANT**: be careful when broadcasting a `Sparse` tensor--the result may not be so sparse!
    For example, broadcasting a `Sparse` tensor with shape [2, 1] with exactly one element into shape [2, 1000]
    will result in a `Sparse` tensor with 1000 elements.

    The `and`, `div`, and `mul` methods are optimized to avoid this issue by ignoring right-hand values at coordinates
    which are not filled in the left-hand `Tensor`.
    """

    __uri__ = uri(Tensor) + "/sparse"

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """
        Return a `Sparse` tensor with the given shape and data type.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.expect(shape, dtype)(ref.Get(cls, (shape, dtype)))

    def elements(self, bounds=None):
        """Return a :class:`Stream` of this tensor's (:class:`Tuple`, :class:`Number`) coordinate-value elements."""

        bounds = _handle_bounds(bounds)
        return self._get("elements", bounds, Stream)

    def as_dense(self):
        """Return a :class:`Dense` view of this `Sparse` tensor."""

        return self._get("dense", rtype=Dense)


def einsum(format, tensors):
    """
    Return the Einstein summation of the given `tensors` according the the given `format` string.

    Example: `einsum("ij,jk->ik", [a, b]) # multiply two matrices`

    The tensor product is computed from left to right, so when using any `Sparse` tensors,
    it's important to put the sparsest first in the list to avoid redundant broadcasting.
    """

    return Tensor(ref.Post(uri(Tensor) + "/einsum", {"format": format, "tensors": tensors}))


def tile(tensor, multiples):
    """Construct a new `Tensor` by tiling the given `tensor` `multiples` times.

    The values of `tensor` are repeated `multiples[x]` times along the `x`th axis of the output.
    `multiples` must be a positive integer or a `Tuple` of length `tensor.ndim`.
    """

    rtype = tensor.__class__ if isinstance(tensor, Tensor) else Tensor
    return rtype(ref.Post(uri(Tensor) + "/tile", {"tensor": tensor, "multiples": multiples}))


def where(cond, x, y):
    """
    Return a view of `x` and `y` depending on whether the corresponding element of `cond` is `True`.

    `cond`, `x`, and `y` must support broadcasting to the same shape.
    """

    return (cond.cast(Bool) * x) + (cond.logical_not() * y)


def _handle_bounds(bounds):
    if bounds is None or isinstance(bounds, ref.Ref) or isinstance(bounds, URI):
        return bounds

    if isinstance(bounds, State):
        form = bounds
        while hasattr(form, "__form__"):
            form = form_of(form)

        if isinstance(form, tuple) or isinstance(form, list):
            bounds = form
        else:
            return bounds

    if hasattr(bounds, "__iter__"):
        bounds = tuple(bounds)
    else:
        bounds = (bounds,)

    return [Range.from_slice(x) if isinstance(x, slice) else x for x in bounds]
