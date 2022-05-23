"""An n-dimensional array of numbers."""

import inspect
import logging
import math

from ...decorators import post
from ...generic import Map, Tuple
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
from ...uri import uri

from ..base import Collection

from .operator import Broadcast, Concatenate, Copy, Expand, Flip, Norm, Read, Reshape, Slice, Sum, Transpose


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
        return Class(ref.Get(ref.MethodSubject(self, "dtype")))

    @property
    def ndim(self):
        return U64(ref.Get(ref.MethodSubject(self, "ndim")))

    @property
    def shape(self):
        return Shape(ref.Get(ref.MethodSubject(self, "shape")))

    @property
    def size(self):
        return U64(ref.Get(ref.MethodSubject(self, "size")))

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

        return ref.Get(ref.MethodSubject(self, "argmax"), axis)

    def broadcast(self, shape):
        """Broadcast this :class:`NDArray` into the given `shape`."""

        return ref.Get(ref.MethodSubject(self, "broadcast"), shape)

    def cast(self, number_type):
        """Cast the data type of this :class:`NDArray` into the given `number_type`."""

        return self._get("cast", number_type)

    def copy(self):
        """Return a copy of this :class:`NDArray`"""

        return ref.Post(uri(Tensor) + "/copy_from", {"tensor": self})

    def expand_dims(self, axis=-1):
        """Expand the given `axis` of this :class:`NDArray`, or append a new axis if no `axis` is given."""

        return ref.Get(ref.MethodSubject(self, "expand_dims"), axis)

    def flip(self, axis):
        """Flip this :class:`NDArray` along the given `axis`"""

        return self._get("flip", axis)

    def max(self, axis=None):
        """
        Find the maxima of this :class:`NDArray` along the given `axis`, or the entire array if no `axis` is given.
        """

        return ref.Get(ref.MethodSubject(self, "max"), axis)

    def min(self, axis=None):
        """
        Find the minima of this :class:`NDArray` along the given `axis`, or the entire array if no `axis` is given.
        """

        return ref.Get(ref.MethodSubject(self, "min"), axis)

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

        return ref.Post(ref.MethodSubject(self, "norm"), _reduce_args(axis, keepdims))

    def product(self, axis=None):
        """
        Calculate the product of this :class:`NDArray` along the given `axis`,
        or the total product if no `axis` is given.
        """

        return ref.Get(ref.MethodSubject(self, "product"), axis)

    def reshape(self, shape):
        """Reshape this :class:`NDArray` into the given `shape`."""

        return ref.Get(ref.MethodSubject(self, "reshape"), shape)

    def slice(self, bounds):
        """Return the sub-array of this :class:`NDArray` within the given `bounds`."""

        return ref.Get(ref.MethodSubject(self), handle_bounds(bounds))

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

        return ref.Post(ref.MethodSubject(self, "sum"), _reduce_args(axis, keepdims))

    def transpose(self, permutation=None):
        """
        Transpose this :class:`NDArray` according to the given `permutation`.

        If no `permutation` is given, this will reverse the order of the axes of this :class:`NDArray`.
        """

        return ref.Get(ref.MethodSubject(self, "transpose"), permutation)

    def write(self, value):
        """Overwrite this :class:`NDArray` with the given :class:`NDArray` or `Number`, broadcasting if needed."""

        return self._put("", None, value)


class Tensor(Collection, NDArray, Trigonometric, Boolean, Numeric, Compare):
    """An n-dimensional array of numbers."""

    __uri__ = uri(Collection) + "/tensor"
    __spec__ = (Shape, Number)

    def __init__(self, form):
        if isinstance(form, Number) or isinstance(deref(form), (bool, float, int)):
            raise ValueError(f"invalid form for Tensor: {form}--consider using a Number instead")

        Collection.__init__(self, form)

    @classmethod
    def trig_rtype(cls):
        shape, dtype = cls.__spec__
        return cls.expect(shape, dtype.trig_rtype())

    @classmethod
    def expect(cls, shape, dtype):
        """
        Define a new subclass of `cls` which captures the given shape and datatype.

        It would be better to implement this feature using generic type parameters (i.e. `class Tensor[Shape, DType]:`)
        but this is not supported on Python <= 3.8.
        """

        if inspect.isclass(shape):
            if shape is not Tuple and shape is not Shape:
                raise ValueError(f"invalid type for tensor shape: {shape}")

        elif not isinstance(shape, (list, tuple, Tuple)):
            raise ValueError(f"invalid tensor shape: {shape}")

        spec = (shape, dtype)

        if not isinstance(shape, Shape):
            shape = Shape(shape)

        class _Tensor(cls):
            __spec__ = spec

            def __init__(self, form):
                if ref.is_literal(shape) and operator(form):
                    try:
                        actual_shape = operator(form).shape
                        if len(shape) != len(actual_shape) or not all(e == a for e, a in zip(shape, actual_shape)):
                            raise ValueError(f"wrong shape for {self}: {actual_shape} (expected {shape})")
                    except (RuntimeError, ValueError) as e:
                        logging.debug(lambda: f"{form} does not have a literal shape: {e}")

                Tensor.__init__(self, form)

            @classmethod
            def create(cls):
                op_ref = ref.Get(cls, (shape, dtype))
                return cls(op_ref)

            @property
            def dtype(self):
                return dtype

            @property
            def ndim(self):
                if hasattr(shape, "__len__"):
                    return len(shape)
                else:
                    return ref.Get(ref.MethodSubject(self, "ndim"))

            @property
            def schema(self):
                return shape, dtype

            @property
            def shape(self):
                return shape

        return _Tensor

    @classmethod
    def create(cls, shape, dtype=F32):
        """Create a new, empty `Tensor`. Call this method to initialize a persistent `Tensor` in a `Chain`."""

        return cls.expect(shape, dtype).create()

    @classmethod
    def load(cls, shape, data, dtype=F32, name=None):
        """
        Load a `Tensor` from an existing data set.

        The `name` parameter is useful in the string representation of an operator graph.

        Example:
            .. highlight:: python
            .. code-block:: python

                coords = [[0, 0, 1], [0, 1, 0]]
                values = [1, 2]
                sparse = tc.tensor.Sparse.load([2, 3, 4], zip(coords, values))
                dense = tc.tensor.Dense.load([2, 3, 4], values, tc.I32)
        """

        name = name if name else f"load {shape}"

        cls = cls.expect(shape, dtype)
        op_ref = ref.Get(uri(cls) + "/load", ((shape, dtype), data), name)
        return cls(op_ref)

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

    @property
    def ndim(self):
        """Return the number of dimensions of this `Tensor`."""

        if hasattr(self.shape, "__len__"):
            return len(self.shape)
        else:
            return self._get("ndim", rtype=U64)

    @property
    def shape(self):
        """Return the shape of this `Tensor`."""

        if operator(self):
            try:
                return operator(self).shape
            except (RuntimeError, ValueError):
                logging.debug(f"{self} does not have a literal shape")

        return self._get("shape", rtype=Shape)

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
        return Tensor(form=NDArray.cast(self, dtype))

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

    def reshape(self, shape, copy=True):
        """Return a view of this `Tensor` with the given `shape`."""

        if not ref.is_literal(copy):
            raise ValueError(f"reshape requires a literal boolean for copy, not {copy}")

        reshaped = Tensor(form=Reshape(self, shape))

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


class Dense(Tensor):
    """
    An n-dimensional array of numbers stored as sequential blocks.

    **IMPORTANT**: for efficiency reasons, serialization of a `Dense` tensor will stop if a non-numeric value
    (NaN or +/- infinity) is encountered. If you receive a `Dense` tensor without enough elements for its shape,
    you can safely treat this response as a divide-by-zero error.
    """

    __uri__ = uri(Tensor) + "/dense"

    @classmethod
    def arange(cls, shape, start, stop, name=None):
        """
        Return a `Dense` tensor with the given shape containing a range of numbers evenly distributed
        between `start` and `stop`.

        The `name` parameter is used only in the string representation of an operator graph.
        """

        name = name if name else f"arange({start}, {stop})x{shape}"
        dtype = type(start) if isinstance(start, Number) else Number
        cls = cls.expect(shape, dtype)
        op_ref = ref.Get(uri(cls) + "/range", (shape, start, stop), name)
        return cls(op_ref)

    @classmethod
    def concatenate(cls, tensors, axis=0):
        """Create a new `Dense` tensor by concatenating the given `tensors` along the given `axis`."""

        return Dense(form=Concatenate(tensors, axis))

    @classmethod
    def constant(cls, shape, value, name=None):
        """
        Return a `Dense` tensor filled with the given `value`.

        The `name` parameter is used only in the string representation of an operator graph.
        """

        name = name if name else f"{value}x{shape}"
        dtype = type(value) if isinstance(value, Number) else Number
        cls = cls.expect(shape, dtype)
        op_ref = ref.Get(uri(cls) + "/constant", (shape, value), name)

        if ref.same_as(value, 1):
            return cls(op_ref)
        if ref.same_as(value, 0):
            return cls(op_ref)
        else:
            return cls(op_ref)

    @classmethod
    def ones(cls, shape):
        """Construct a `Dense` tensor with dtype :class:`F64` filled with ones."""

        return cls.expect(shape, F64).constant(shape, 1.)

    @classmethod
    def ones_like(cls, tensor):
        """Return a `Dense` tensor filled with ones, with the same shape and data type as the given `tensor`."""

        return cls.ones(tensor.shape)

    @classmethod
    def zeros(cls, shape):
        """Construct a `Dense` tensor with dtype :class:`F64` filled with ones."""

        return cls.expect(shape, F64).constant(shape, 0.)

    @classmethod
    def zeros_like(cls, tensor):
        """Return a `Dense` tensor filled with zeros, with the same shape and data type as the given `tensor`."""

        return cls.zeros(tensor.shape)

    @classmethod
    def random_normal(cls, shape, mean=0.0, std=1.0, name=None):
        """
        Return a `Dense` tensor filled with a random normal distribution of `F64` s.

        The `name` parameter is used only in the string representation of an operator graph.
        """

        name = name if name else f"random {shape}"
        cls = cls.expect(shape, F64)
        args = {"shape": shape, "mean": mean, "std": std}
        op_ref = ref.Post(uri(cls) + "/random/normal", args, name)
        return cls(op_ref)

    @classmethod
    def random_uniform(cls, shape, minval=0, maxval=1):
        """Return a `Dense` tensor filled with a uniform random distribution of `F64` s."""

        if not ref.is_literal(minval) or not ref.is_literal(maxval):
            raise ValueError(f"Dense.random_uniform requires a literal range, not [{minval}, {maxval})")

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
        return cls(truncated)

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
        return Dense(form=Tensor.sub(self, other))


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
    def zeros(cls, shape, dtype=F32, name=None):
        """
        Return a `Sparse` tensor with the given shape and data type.

        If `dtype` is not specified, the data type will be :class:`F32`.

        The `name` parameter is used only in the string representation of an operator graph.
        """

        cls = cls.expect(shape, dtype)
        name = name if name else f"sparse 0x{shape}"
        return cls(ref.Get(cls, (shape, dtype), name))

    @classmethod
    def zeros_like(cls, tensor):
        """Return a `Sparse` tensor with the same shape and data type as the given `tensor`."""

        return cls.zeros(tensor.shape)

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
