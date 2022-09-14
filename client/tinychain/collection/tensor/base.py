"""An n-dimensional array of numbers."""

import inspect
import math
import typing

from ...decorators import post
from ...generic import autobox, gcs, resolve_class, Map
from ...interface import Compare, Interface
from ...math.interface import Boolean, Numeric, Trigonometric
from ...math.operator import deref, is_one, is_zero, operator
from ...math.operator import Abs, Exp, Log, Pow
from ...math.operator import Add, Mul, MatMul, Div, Sub
from ...math.operator import Sin, Sinh, Asin, Asinh, Cos, Cosh, Acos, Acosh, Tan, Tanh, Atan, Atanh
from ...math.operator import LogicalAnd, LogicalNot, LogicalOr, LogicalXor
from ...scalar.bound import handle_bounds
from ...scalar.number import Bool, F32, F64, Number, U64
from ...scalar import ref
from ...shape import Shape
from ...state import Class, State, Stream
from ...uri import URI

from ..base import Collection

from .operator import Broadcast, Cast, Concatenate, Copy, Expand, Flip, Read, Reshape, Slice, Transpose
from .operator import Max, Min, Norm, Product, Sum


DType = typing.TypeVar("DType", bound=type[Number])


class NDArray(Interface):
    def __getitem__(self, bounds):
        return self.slice(bounds)

    def __matmul__(self, other):
        from .functions import einsum
        return einsum("...ij,...jk->ik", [self, other])

    def __setitem__(self, bounds, value):
        raise NotImplementedError("use Tensor.write instead")

    @property
    def dtype(self):
        return Class(ref.Get(URI(self, "dtype")))

    @property
    def ndim(self):
        return U64(ref.Get(URI(self, "ndim")))

    @property
    def shape(self):
        return Shape(ref.Get(URI(self, "shape")))

    @property
    def size(self):
        return U64(ref.Get(URI(self, "size")))

    def all(self):
        """Return `True` if all elements in this :class:`NDArray` are nonzero."""

        return self._get("all", rtype=Bool)

    def any(self):
        """Return `True` if any element in this :class:`NDArray` is nonzero."""

        return self._get("any", rtype=Bool)

    def argmax(self, axis=None):
        """
        Return the indices of the maximum values along the given `axis` of this :class:`NDArray`.

        If no `axis` is given, the total offset will be returned.
        """

        return ref.Get(URI(self, "argmax"), axis)

    def broadcast(self, shape):
        """Broadcast this :class:`NDArray` into the given `shape`."""

        return ref.Get(URI(self, "broadcast"), shape)

    def cast(self, number_type):
        """Cast the data type of this :class:`NDArray` into the given `number_type`."""

        return self._get("cast", number_type)

    def copy(self):
        """Return a copy of this :class:`NDArray`"""

        return ref.Post(URI(Tensor) + "/copy_from", {"tensor": self})

    def expand_dims(self, axis=-1):
        """Expand the given `axis` of this :class:`NDArray`, or append a new axis if no `axis` is given."""

        return ref.Get(URI(self, "expand_dims"), axis)

    def flip(self, axis):
        """Flip this :class:`NDArray` along the given `axis`"""

        return self._get("flip", axis)

    def max(self, axis=None):
        """
        Find the maxima of this :class:`NDArray` along the given `axis`, or the entire array if no `axis` is given.
        """

        return ref.Get(URI(self, "max"), axis)

    def min(self, axis=None):
        """
        Find the minima of this :class:`NDArray` along the given `axis`, or the entire array if no `axis` is given.
        """

        return ref.Get(URI(self, "min"), axis)

    def mean(self, axis=None):
        """
        Return the average of this :class:`NDArray` along the given `axis`, or the total average if no `axis` is given.
        """

        if axis is None:
            return self.sum() / self.size
        else:
            return self.sum(axis) / self.shape[axis]

    def logical_and(self, other):
        return Tensor(form=LogicalAnd(self, other))

    def logical_not(self):
        if is_zero(self):
            return Dense.ones_like(self)
        elif is_one(self):
            return Sparse.zeros_like(self)

        return Tensor(form=LogicalNot(self))

    def logical_or(self, other):
        return Tensor(form=LogicalOr(self, other))

    def logical_xor(self, other):
        return Tensor(form=LogicalXor(self, other))

    def norm(self, axis=None, keepdims=False):
        """
        Compute the Frobenius (aka Euclidean) norm of this :class:`NDArray`.

        By default this is the matrix norm of the last two dimensions; if an `axis` is given,
        it will be the vector norm along that `axis`.
        """

        return ref.Post(URI(self, "norm"), _reduce_args(axis, keepdims))

    def product(self, axis=None):
        """
        Calculate the product of this :class:`NDArray` along the given `axis`,
        or the total product if no `axis` is given.
        """

        return ref.Get(URI(self, "product"), axis)

    def reshape(self, shape):
        """Reshape this :class:`NDArray` into the given `shape`."""

        return ref.Get(URI(self, "reshape"), shape)

    def slice(self, bounds):
        """Return the sub-array of this :class:`NDArray` within the given `bounds`."""

        return ref.Get(URI(self), handle_bounds(bounds))

    def std(self, axis=None):
        """
        Return the standard deviation of this :class:`NDArray` along the given `axis`,
        or the total standard deviation if no `axis` is given.
        """

        if axis is None:
            average = self.mean()
            size = self.size
            return (((self - average)**2).sum() / size)**0.5
        else:
            raise NotImplementedError("std with axis")

    def sum(self, axis=None, keepdims=False):
        """Compute the sum of this :class:`NDArray` along the given `axis`, or the total sum if no `axis` is given."""

        return ref.Post(URI(self, "sum"), _reduce_args(axis, keepdims))

    def transpose(self, permutation=None):
        """
        Transpose this :class:`NDArray` according to the given `permutation`.

        If no `permutation` is given, this will reverse the order of the axes of this :class:`NDArray`.
        """

        return ref.Get(URI(self, "transpose"), permutation)

    def write(self, value):
        """Overwrite this :class:`NDArray` with the given :class:`NDArray` or `Number`, broadcasting if needed."""

        return self._put("", None, value)


class Tensor(Collection, NDArray, Trigonometric, Boolean, Numeric, Compare, typing.Generic[DType]):
    """An n-dimensional array of numbers."""

    __uri__ = URI(Collection) + "/tensor"

    def __init__(self, form):
        if isinstance(form, Number) or isinstance(deref(form), (bool, float, int)):
            raise ValueError(f"invalid form for Tensor: {form}--consider using a Number instead")

        Collection.__init__(self, form)

    def __repr__(self):
        if operator(self):
            return repr(operator(self))
        elif isinstance(deref(self), ref.Op):
            return repr(deref(self))

        return State.__repr__(self)

    def __matmul__(self, other):
        if is_zero(self) or is_zero(other):
            return Sparse.zeros([self.shape[-2], other.shape[-1]])

        return Tensor(form=MatMul(self, other))

    @classmethod
    def trig_rtype(cls):
        rtype = Number

        if hasattr(cls, "__orig_class__"):
            dtype = resolve_class(typing.get_args(cls.__orig_class__)[0])
            rtype = dtype.trig_rtype() if inspect.isclass(dtype) and issubclass(dtype, Number) else Number

        return Tensor[rtype]

    @classmethod
    def with_shape(cls, shape):
        if not hasattr(shape, "__len__"):
            return cls

        class TensorWithExpectedShape(cls):
            @property
            def ndim(self):
                return len(shape)

            @property
            def shape(self):
                default = self._get("shape", rtype=Shape)
                return Shape([shape[x] if ref.is_literal(shape[x]) else default[x] for x in range(len(shape))])

        return TensorWithExpectedShape

    @property
    def dtype(self):
        if hasattr(self, "__orig_class__"):
            _shape, dtype = typing.get_args(self.__orig_class__)
            dtype = resolve_class(dtype)
            if inspect.isclass(dtype) and issubclass(dtype, Number):
                return dtype

        return Class[Number](form=ref.Get(URI(self, "dtype")))

    @property
    def ndim(self):
        """Return the number of dimensions of this `Tensor`."""

        shape = self.shape

        if hasattr(shape, "__len__"):
            return len(shape)
        else:
            return self._get("ndim", rtype=U64)

    @property
    def schema(self):
        return self.shape, self.dtype

    @property
    def shape(self):
        """Return the shape of this `Tensor`."""

        default = self._get("shape", rtype=Shape)

        try:
            return getattr(ref.deref(self), "shape", default)
        except (RuntimeError, ValueError):
            pass

        return default

    def abs(self):
        return Tensor(form=Abs(self))

    def add(self, other):
        if ref.same_as(other, 0):
            return self

        return Tensor(form=Add(self, other))

    def broadcast(self, shape):
        if ref.same_as(shape, self.shape):
            return self

        return Tensor(form=Broadcast(self, shape))

    def cast(self, dtype):
        try:
            return Tensor[dtype](form=Cast(self, dtype))
        except TypeError:
            return Tensor(form=Cast(self, dtype))

    def copy(self):
        return Tensor(form=Copy(self))

    def div(self, other):
        if ref.same_as(other, 1):
            return self

        return Tensor(form=Div(self, other))

    def exp(self):
        if is_one(self):
            return Dense.constant(self.shape, math.e, self.dtype)
        elif is_zero(self):
            return Sparse.zeros_like(self)

        return Tensor(form=Exp(self))
    
    def log(self, base=None):
        return Tensor(form=Log(self, base))

    def max(self, axis=None):
        return Tensor(form=NDArray.max(self, axis))

    def min(self, axis=None):
        return Tensor(form=NDArray.min(self, axis))

    def sin(self):
        return Tensor(form=Sin(self))

    def cos(self):
        return Tensor(form=Cos(self))

    def asin(self):
        return Tensor(form=Asin(self))

    def acos(self):
        return Tensor(form=Acos(self))

    def sinh(self):
        return Tensor(form=Sinh(self))

    def cosh(self):
        return Tensor(form=Cosh(self))

    def asinh(self):
        return Tensor(form=Asinh(self))

    def acosh(self):
        return Tensor(form=Acosh(self))

    def tan(self):
        return Tensor(form=Tan(self))

    def tanh(self):
        return Tensor(form=Tanh(self))

    def atan(self):
        return Tensor(form=Atan(self))

    def atanh(self):
        return Tensor(form=Atanh(self))

    def flip(self, axis):
        """Flip the elements in this `Tensor` along the specified `axis`."""

        return Tensor(form=Flip(self, axis))

    def eq(self, other):
        """Return a boolean `Tensor` with element-wise equality values."""

        return self._post("eq", {"r": other}, Tensor)

    def expand_dims(self, axis=None):
        """Return a view of this `Tensor` with an extra dimension of size 1 at the given axis."""

        return Tensor(form=Expand(self, axis))

    def gt(self, other):
        """Return a boolean `Tensor` with element-wise greater-than values."""

        return self._post("gt", {"r": other}, Tensor)

    def gte(self, other):
        """Return a boolean `Tensor` with element-wise greater-or-equal values."""

        return self._post("gte", {"r": other}, Tensor)

    def lt(self, other):
        """Return a boolean `Tensor` with element-wise less-than values."""

        return self._post("lt", {"r": other}, Tensor)

    def lte(self, other):
        """Return a boolean `Tensor` with element-wise less-or-equal values."""

        return self._post("lte", {"r": other}, Tensor)

    def max(self, axis=None, keepdims=False):
        """
        Return the maximum value of this `Tensor` along the given `axis`, or the overall maximum if no axis is given.
        """

        rtype = Number if axis is None else Tensor
        return rtype(Max(self, axis, keepdims))

    def min(self, axis=None, keepdims=False):
        """
        Return the minimum value of this `Tensor` along the given `axis`, or the overal minimum if no axis is given.
        """

        rtype = Number if axis is None else Tensor
        return rtype(Min(self, axis, keepdims))

    def mul(self, other):
        if ref.same_as(other, 1):
            return self

        return Tensor(form=Mul(self, other))

    def ne(self, other):
        """Return a boolean `Tensor` with element-wise not-equal values."""

        return self._post("ne", {"r": other}, Tensor)

    def norm(self, axis=None, keepdims=False):
        """
        With no `axis`, computes the Frobenius norm (aka Euclidean norm) of a matrix or batch of matrices.

        For a vector norm, specify the `axis` of the vector.
        """

        return Tensor(form=Norm(self, axis, keepdims))

    def pow(self, other):
        if ref.same_as(other, 1):
            return self

        return Tensor(form=Pow(self, other))

    def product(self, axis=None, keepdims=False):
        """Calculate the product of this `Tensor` along the given `axis`, or the total product if no axis is given."""

        rtype = Number if axis is None else Tensor
        return rtype(Product(self, axis, keepdims))

    def reshape(self, shape, copy=True):
        """Return a view of this `Tensor` with the given `shape`."""

        if not ref.is_literal(copy):
            raise ValueError(f"reshape requires a literal boolean for copy, not {copy}")

        reshaped = Tensor(form=Reshape(self, Shape(shape)))

        if copy:
            return reshaped.copy()
        else:
            return reshaped

    def slice(self, bounds):
        parent = self
        bounds = handle_bounds(bounds)

        if ref.is_literal(self.shape):
            slice_shape = self.shape.slice(bounds)  # test for valid bounds, if possible
            if hasattr(slice_shape, "__len__") and len(slice_shape) == 0:
                # in this case the result is a Number, not a Tensor
                rtype = self.dtype if inspect.isclass(self.dtype) and issubclass(self.dtype, Number) else Number

                class WritableView(rtype):
                    def write(self, value):
                        return parent._put("", bounds, value)

                return WritableView(Read(self, bounds))

        class WritableView(Tensor):
            def write(self, value):
                return parent._put("", bounds, value)

        return WritableView(Slice(self, bounds))

    def sub(self, other):
        if ref.same_as(other, 0):
            return self

        return Tensor(form=Sub(self, other))

    def sum(self, axis=None, keepdims=False):
        """Calculate the sum of this `Tensor` along the given `axis`, or the total sum if no axis is given."""

        rtype = Number if axis is None else Tensor
        return rtype(Sum(self, axis, keepdims))

    def transpose(self, permutation=None):
        """
        Return a view of this `Tensor` with its axes transposed according to the given permutation.

        If no permutation is given, the axes will be inverted (e.g. `(0, 1, 2)` inverts to `(2, 1, 0)`).
        """

        return Tensor(form=Transpose(self, permutation))


class Dense(Tensor, typing.Generic[DType]):
    """
    An n-dimensional array of numbers stored as sequential blocks.

    **IMPORTANT**: for efficiency reasons, serialization of a `Dense` tensor will stop if a non-numeric value
    (NaN or +/- infinity) is encountered. If you receive a `Dense` tensor without enough elements for its shape,
    you can safely treat this response as a divide-by-zero error.
    """

    __uri__ = URI(Tensor) + "/dense"

    @classmethod
    def arange(cls, shape, start, stop):
        """
        Return a `Dense` tensor with the given shape containing a range of numbers evenly distributed
        between `start` and `stop`.
        """

        return cls.with_shape(shape)(form=ref.Get(URI(cls, "range"), (shape, start, stop)))

    @classmethod
    def concatenate(cls, tensors, axis=0):
        """Create a new `Dense` tensor by concatenating the given `tensors` along the given `axis`."""

        return cls(form=Concatenate(tensors, axis))

    @classmethod
    def create(cls, shape, dtype=F32):
        """
        Create a new, empty :class:`Dense` tensor.

        Call this method to initialize a persistent `Tensor` in a `Chain`.
        """

        return cls[dtype].with_shape(shape)(form=ref.Get(URI(cls), (shape, dtype)))

    @classmethod
    def load(cls, shape, data, dtype=F32):
        """
        Load a :class:`Dense` tensor from an existing data set.

        Example:
            .. highlight:: python
            .. code-block:: python

                values = [0, 1, 2]
                dense = tc.tensor.Dense.load([1, 3], values, tc.I32)
        """

        return cls[dtype].with_shape(shape)(form=ref.Get(URI(cls, "load"), ((shape, dtype), data)))

    @classmethod
    def constant(cls, shape, value):
        """Return a `Dense` tensor of the given `shape` filled with the given `value`."""

        assert not inspect.isclass(value)
        value = autobox(value)
        op_ref = ref.Get(URI(cls, "constant"), (shape, value))

        if isinstance(value, Number):
            try:
                return cls[type(value)].with_shape(shape)(form=op_ref)
            except TypeError:
                pass

        return cls.with_shape(shape)(form=op_ref)

    @classmethod
    def ones(cls, shape, dtype=F32):
        """Construct a `Dense` tensor filled with ones."""

        try:
            return cls[dtype].constant(shape, dtype(1.))
        except TypeError:
            return cls.constant(shape, 1.)

    @classmethod
    def ones_like(cls, tensor):
        """Return a `Dense` tensor filled with ones, with the same shape as the given `tensor`."""

        # TODO: include data type
        return cls.ones(tensor.shape)

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """Construct a `Dense` tensor filled with zeros."""

        try:
            return cls[dtype].constant(shape, dtype(0.))
        except TypeError:
            return cls.constant(shape, 0)

    @classmethod
    def zeros_like(cls, tensor):
        """Return a `Dense` tensor filled with zeros, with the same shape as the given `tensor`."""

        # TODO: include data type
        return cls.zeros(tensor.shape)

    @classmethod
    def random_normal(cls, shape, mean=0.0, std=1.0):
        """Return a `Dense` tensor filled with a random normal distribution of `F64` s."""

        class RandomNormal(cls[F64]):
            @property
            def shape(self):
                default = self._get("shape", rtype=Shape)

                if hasattr(shape, "__len__"):
                    return Shape([shape[x] if ref.is_literal(shape[x]) else default[x] for x in range(len(shape))])
                else:
                    return default

        args = {"shape": shape, "mean": mean, "std": std}
        op_ref = ref.Post(URI(cls, "random/normal"), args)
        return RandomNormal(form=op_ref)

    @classmethod
    def random_uniform(cls, shape, minval=0, maxval=1):
        """Return a `Dense` tensor filled with a uniform random distribution of `F64` s."""

        if not ref.is_literal(minval) or not ref.is_literal(maxval):
            raise ValueError(f"Dense.random_uniform requires a literal range, not [{minval}, {maxval})")

        if minval == maxval:
            return cls[F64].constant(shape, float(minval))

        assert maxval > minval

        random = cls[F64].with_shape(shape)(form=ref.Get(URI(cls, "random", "uniform"), shape))

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

        if not ref.is_literal((mean, std)):
            raise TypeError(f"truncated normal distribution requires literal parameters, not mean={mean} and std={std}")

        if not std:
            return cls[F64].constant(shape, float(mean))

        minval = std * -2 if minval is None else minval
        maxval = std * 2 if maxval is None else maxval

        @post
        def cond(cxt, tensor: Dense) -> Bool:
            cxt.too_small = (tensor < minval).any()
            cxt.too_big = (tensor > maxval).any()
            return cxt.too_small.logical_or(cxt.too_big)

        @post
        def step(cxt, tensor: Dense) -> Map:
            from .functions import where
            cxt.new_dist = Dense.random_normal(shape, mean, std)
            cxt.new_tensor = where((tensor >= minval).logical_and(tensor <= maxval), tensor, cxt.new_dist)
            return Map(tensor=cxt.new_tensor.copy())

        truncated = Map(ref.While(cond, step, Map(tensor=Dense.random_normal(shape, mean, std))))["tensor"]
        return cls[F64](form=truncated)

    def add(self, other):
        if ref.same_as(other, 0):
            return self

        return Dense(form=Tensor.add(self, other))

    def argsort(self):
        """Return the coordinates needed to sort this `Tensor`."""

        return self._get("argsort", rtype=Tensor)

    def as_sparse(self):
        """Return a :class:`Sparse` view of this `Dense` tensor."""

        return self._get("sparse", rtype=Sparse)

    def elements(self, bounds=None):
        """Return a :class:`Stream` of the :class:`Number` elements of this `Dense` tensor."""

        bounds = handle_bounds(bounds)
        return self._get("elements", bounds, Stream)

    def sub(self, other):
        if ref.same_as(other, 0):
            return self

        return Dense(form=Tensor.sub(self, other))


class Sparse(Tensor, typing.Generic[DType]):
    """
    An n-dimensional array of numbers stored as a :class:`Table` of coordinates and values.

    **IMPORTANT**: be careful when broadcasting a `Sparse` tensor--the result may not be so sparse!
    For example, broadcasting a `Sparse` tensor with shape [2, 1] with exactly one element into shape [2, 1000]
    will result in a `Sparse` tensor with 1000 elements.

    The `and`, `div`, and `mul` methods are optimized to avoid this issue by ignoring right-hand values at coordinates
    which are not filled in the left-hand `Tensor`.
    """

    __uri__ = URI(Tensor) + "/sparse"

    @classmethod
    def create(cls, shape, dtype=F32):
        """
        Create a new, empty :class:`Sparse` tensor.

        Call this method to initialize a persistent :class:`Tensor` in a :class:`Chain`.
        """

        op_ref = ref.Get(URI(cls), (shape, dtype))

        try:
            return cls[dtype].with_shape(shape)(form=op_ref)
        except TypeError:
            return cls.with_shape(shape)(form=op_ref)

    @classmethod
    def load(cls, shape, data, dtype=F32):
        """
        Load a :class:`Sparse` tensor from an existing data set.

        Example:
            .. highlight:: python
            .. code-block:: python

                coords = [[0, 0, 1], [0, 1, 0]]
                values = [1, 2]
                sparse = tc.tensor.Sparse.load([2, 3, 4], zip(coords, values))
        """

        op_ref = ref.Get(URI(cls, "load"), ((shape, dtype), data))

        try:
            return cls[dtype].with_shape(shape)(form=op_ref)
        except TypeError:
            return cls.with_shape(shape)(form=op_ref)

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """
        Return a `Sparse` tensor with the given shape and data type.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        op_ref = ref.Get(URI(cls), (shape, dtype))

        try:
            return cls[dtype].with_shape(shape)(form=op_ref)
        except TypeError:
            return cls.with_shape(shape)(form=op_ref)

    @classmethod
    def zeros_like(cls, tensor):
        """Return a `Sparse` tensor with the same shape and data type as the given `tensor`."""

        return cls[tensor.dtype].zeros(tensor.shape, tensor.dtype)

    def as_dense(self):
        """Return a :class:`Dense` view of this `Sparse` tensor."""

        return self._get("dense", rtype=Dense)

    def div(self, other):
        if ref.same_as(other, 1):
            return self

        return Sparse(form=Div(self, other))

    def elements(self, bounds=None):
        """
        Return a :class:`Stream` of this tensor's `(coord, number)` coordinate-value elements.

        `coord` is a :class:`Tuple` of :class:`U64` coordinates, and `number` is the element at `coord`.
        """

        bounds = handle_bounds(bounds)
        return self._get("elements", bounds, Stream)

    def mul(self, other):
        if ref.same_as(other, 1):
            return self

        return Sparse(form=Tensor.mul(self, other))


def _reduce_args(axis=None, keepdims=False):
    args = {}

    if axis is not None:
        args["axis"] = axis

    if keepdims:
        args["keepdims"] = keepdims

    return args
