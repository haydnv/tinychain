"""An n-dimensional array of numbers."""

import inspect
import itertools
import logging

from ..decorators import post
from ..generic import Map, Tuple
from ..interface import Compare, Interface
from ..math.operator import *
from ..scalar.bound import handle_bounds
from ..scalar.number import Bool, F32, F64, Number, UInt
from ..scalar import ref
from ..shape import Shape
from ..state import Class, State, StateRef, Stream
from ..uri import uri

from .base import Collection


class NDArray(Interface):
    @property
    def dtype(self):
        return Class(ref.Get(ref.MethodSubject(self, "dtype")))

    @property
    def shape(self):
        return Shape(ref.Get(ref.MethodSubject(self, "shape")))

    def broadcast(self, shape):
        return ref.Get(ref.MethodSubject(self, "broadcast"), shape)

    def expand_dims(self, axis=-1):
        return ref.Get(ref.MethodSubject(self, "expand_dims"), axis)

    def flip(self, axis):
        return ref.Get(ref.MethodSubject(self, "flip"), axis)

    def norm(self, axis=None, keepdims=False):
        return ref.Post(ref.MethodSubject(self, "norm"), _reduce_args(axis, keepdims))

    def reshape(self, shape):
        return ref.Get(ref.MethodSubject(self, "reshape"), shape)

    def slice(self, bounds):
        return ref.Get(ref.MethodSubject(self), bounds)

    def sum(self, axis=None, keepdims=False):
        return ref.Post(ref.MethodSubject(self, "sum"), _reduce_args(axis, keepdims))

    def transpose(self, permutation=None):
        return ref.Get(ref.MethodSubject(self, "transpose"), permutation)


class Tensor(Collection, Numeric, Compare, Trigonometric, NDArray):
    """An n-dimensional array of numbers."""

    __uri__ = uri(Collection) + "/tensor"
    __spec__ = (Shape, Number)

    def __init__(self, form):
        if isinstance(deref(form), (bool, float, int)):
            raise ValueError(f"invalid form for Tensor: {form}--consider using a Number instead")

        Collection.__init__(self, form)

    def __repr__(self):
        if operator(self):
            return str(operator(self))

        return State.__repr__(self)

    @classmethod
    def trig_rtype(cls):
        shape, dtype = cls.__spec__
        return cls.expect(shape, dtype.trig_rtype())

    @classmethod
    def expect(cls, shape, dtype):
        """
        Define a new subclass of `cls` which captures the given shape and datatype.

        It would be better to implement this function using generic type parameters (i.e. `class Tensor[Shape, DType]:`)
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
        op_ref = ref.Get(uri(cls) + "/load", ((shape, dtype), data))
        return cls(op_ref)

    def __getitem__(self, bounds):
        return self.slice(bounds)

    def __setitem__(self, bounds, value):
        raise NotImplementedError("use Tensor.write instead")

    def __matmul__(self, other):
        return Tensor(MatMul(self, other))
    
    def abs(self):
        return Tensor(Abs(self))

    def add(self, other):
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

    def broadcast(self, shape):
        return self.__class__(form=Broadcast(self, shape))

    def cast(self, number_type):
        """Cast the data type of `Tensor` into the given `number_type`."""

        return self._get("cast", number_type, self.__class__.expect(self.shape, number_type))

    def copy(self):
        """Return a copy of this `Tensor`"""

        return self.__class__(form=Copy(self))

    def div(self, other):
        return Tensor(Div(self, other))

    @property
    def dtype(self):
        """Return the data type of this `Tensor`."""

        return self._get("dtype", rtype=Class)

    def exp(self):
        return Tensor(Exp(self))
    
    def log(self, base=None):
        return Tensor(Log(self, base))

    def sin(self):
        return Tensor(Sin(self))

    def cos(self):
        return Tensor(Cos(self))

    def asin(self):
        return Tensor(Asin(self))

    def acos(self):
        return Tensor(Acos(self))

    def sinh(self):
        return Tensor(Sinh(self))

    def cosh(self):
        return Tensor(Cosh(self))

    def asinh(self):
        return Tensor(Asinh(self))

    def acosh(self):
        return Tensor(Acosh(self))

    def tan(self):
        return Tensor(Tan(self))

    def tanh(self):
        return Tensor(Tanh(self))

    def atan(self):
        return Tensor(Atan(self))

    def atanh(self):
        return Tensor(Atanh(self))

    def flip(self, axis):
        """Flip the elements in this `Tensor` along the specified `axis`."""

        return self.__class__(form=Flip(self, axis))

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

    def logical_and(self, other):
        """Return a boolean `Tensor` with element-wise logical and values."""

        return self._post("and", {"r": other}, Tensor)

    def logical_not(self):
        """Return a boolean `Tensor` with element-wise logical not values."""

        return self._get("not", rtype=Tensor)

    def logical_or(self, other):
        """Return a boolean `Tensor` with element-wise logical or values."""

        return self._post("or", {"r": other}, Tensor)

    def logical_xor(self, other):
        """Return a boolean `Tensor` with element-wise logical xor values."""

        return self._post("xor", {"r": other}, Tensor)

    def max(self, axis=None):
        """Find the maxima of this `Tensor` along the given `axis`, or the entire tensor if no axis is given."""

        rtype = Number if axis is None else Tensor
        return self._get("max", axis, rtype)

    def min(self, axis=None):
        """Find the minima of this `Tensor` along the given `axis`, or the entire tensor if no axis is given."""

        rtype = Number if axis is None else Tensor
        return self._get("min", axis, rtype)

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
        return Tensor(Mul(self, other))

    @property
    def ndim(self):
        """Return the number of dimensions of this `Tensor`."""

        return self._get("ndim", rtype=UInt)

    def ne(self, other):
        """Return a boolean `Tensor` with element-wise not-equal values."""

        return self._post("ne", {"r": other}, Tensor)

    def norm(self, axis=None):
        """
        With no `axis`, computes the Frobenius norm (aka Euclidean norm) of a matrix or batch of matrices.

        For a vector norm, specify the `axis` of the vector.
        """

        return Tensor(Norm(self, axis))

    def pow(self, other):
        return Tensor(Pow(self, other))

    def product(self, axis=None):
        """Calculate the product of this `Tensor` along the given `axis`, or the total product if no axis is given."""

        rtype = Number if axis is None else Tensor
        return self._get("product", axis, rtype)

    def reshape(self, shape, copy=True):
        """Return a view of this `Tensor` with the given `shape`."""

        if not ref.is_literal(copy):
            raise ValueError(f"reshape requires a constant boolean for copy, not {copy}")

        reshaped = Tensor(form=Reshape(self, shape))

        if copy:
            return reshaped.copy()
        else:
            return reshaped

    @property
    def shape(self):
        """Return the shape of this `Tensor`."""

        if operator(self):
            try:
                return operator(self).shape
            except (RuntimeError, ValueError):
                logging.debug(f"shape of {self} is not constant")
        elif hasattr(deref(self), "shape"):
            return deref(self).shape

        return self._get("shape", rtype=Shape)

    @property
    def size(self):
        """Return the size of this `Tensor` (the product of its `shape`)."""

        return self._get("size", rtype=UInt)

    def slice(self, bounds):
        parent = self
        bounds = handle_bounds(bounds)

        if ref.is_literal(self.shape):
            self.shape.slice(bounds)  # test for valid bounds, if possible

        class WritableView(Tensor):
            def write(self, value):
                return parent._put("", bounds, value)

        return WritableView(Slice(self, bounds))

    def split(self, num_or_size_splits, axis=0):
        """
        Split this `Tensor` into multiple slices along the given `axis`.

        This method requires a constant `num_or_size_splits`, `axis`, and `self.shape[axis]`.

        If `num_or_size_splits` is a `Number`, the `tensor` will be sliced along `axis` `num_or_size_splits` times;
        if `self.shape[axis] % num_or_size_splits != 0` then a `ValueError` error will be raised.

        If `num_or_size_splits` is a `Tuple` with length `n` then the `tensor` will be split into `n` new `Tensor` s
        each with `shape[axis] == num_or_size_splits[axis]`; if the sum of `num_or_size_splits` is not equal to
        `self.shape[axis]` then a `ValueError` error will be raised.
        """

        num_or_size_splits = ref.form_of(num_or_size_splits)
        if not ref.is_literal(num_or_size_splits):
            raise ValueError(f"Tensor.split requires a constant num_or_size_splits, not {num_or_size_splits}")

        if not ref.is_literal(axis):
            raise ValueError(f"Tensor.split requires a constant axis, not {axis}")

        if ref.is_literal(self.shape[axis]):
            dim = ref.form_of(self.shape[axis])
        else:
            raise RuntimeError(f"to split {self} requires a constant dimension to split, not {self.shape[axis]}")

        if isinstance(num_or_size_splits, (list, tuple)):
            if sum(num_or_size_splits) != dim:
                raise ValueError(f"{num_or_size_splits} does not match the dimension {dim} of axis {axis}")

        elif int(num_or_size_splits) == num_or_size_splits:
            if dim % num_or_size_splits != 0:
                raise ValueError(f"split dimension {dim} is not divisible by {num_or_size_splits}")

            slice_dim = dim // num_or_size_splits
            num_or_size_splits = [slice_dim] * num_or_size_splits

        else:
            raise ValueError(f"invalid num_or_size_splits: {num_or_size_splits}")

        start = 0
        slices = []
        for slice_dim in num_or_size_splits:
            bounds = ([slice(None)] * axis) + [slice(start, start + slice_dim)]
            slices.append(self[bounds])

        return slices

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
        return Tensor(Sub(self, other))

    def sum(self, axis=None, keepdims=False):
        """Calculate the sum of this `Tensor` along the given `axis`, or the total sum if no axis is given."""

        rtype = Number if axis is None else Tensor
        return rtype(form=Sum(self, axis, keepdims))

    def transpose(self, permutation=None):
        """
        Return a view of this `Tensor` with its axes transposed according to the given permutation.

        If no permutation is given, the axes will be inverted (e.g. `(0, 1, 2)` inverts to `(2, 1, 0)`).
        """

        return Tensor(form=Transpose(self, permutation))

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
    def arange(cls, shape, start, stop, name=None):
        """
        Return a `Dense` tensor with the given shape containing a range of numbers evenly distributed
        between `start` and `stop`.

        The `name` parameter is used only in the string representation of an operator graph.
        """

        name = name if name else f"arange({start}, {stop})x{shape}"
        dtype = type(start) if isinstance(start, Number) else Number
        cls = cls.expect(shape, dtype)
        op_ref = ref.Get(uri(cls) + "/range", (shape, start, stop))
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
        op_ref = ref.Get(uri(cls) + "/constant", (shape, value))

        if same_as(value, 1):
            return cls(op_ref)
        if same_as(value, 0):
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
        op_ref = ref.Post(uri(cls) + "/random/normal", args)
        return cls(op_ref)

    @classmethod
    def random_uniform(cls, shape, minval=0, maxval=1):
        """Return a `Dense` tensor filled with a uniform random distribution of `F64` s."""

        if not is_literal(minval) or not is_literal(maxval):
            raise ValueError(f"Dense.random_uniform requires a constant range, not [{minval}, {maxval})")

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
    def truncated_normal(cls, shape, mean=0.0, std=1.0, minval=None, maxval=None, name=None):
        """
        Return a `Dense` tensor filled with a random normal distribution of `F64` s.

        Any value `x` outside the range `minval <= x <= maxval` will be replaced by a value drawn from a new
        random normal distribution.

        `minval` and `maxval` default to two standard deviations.

        The `name` parameter is used only in the string representation of an operator graph.
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
            cxt.new_dist = Dense.random_normal(shape, mean, std)
            cxt.new_tensor = where((tensor >= minval).logical_and(tensor <= maxval), tensor, cxt.new_dist)
            return Map(tensor=cxt.new_tensor.copy())

        name = name if name else f"truncated random {shape}"
        truncated = Map(ref.While(cond, step, Map(tensor=Dense.random_normal(shape, mean, std))))["tensor"]
        return cls(truncated)

    def add(self, other):
        return Dense(Add(self, other))

    def argsort(self):
        """Return the coordinates needed to sort this `Tensor`."""

        return self._get("argsort", rtype=self.__class__)

    def as_sparse(self):
        """Return a :class:`Sparse` view of this `Dense` tensor."""

        return self._get("sparse", rtype=Sparse)

    def elements(self, bounds=None):
        """Return a :class:`Stream` of the :class:`Number` elements of this `Dense` tensor."""

        bounds = handle_bounds(bounds)
        return self._get("elements", bounds, Stream)

    def sub(self, other):
        return Dense(Sub(self, other))


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
        return cls(ref.Get(cls, (shape, dtype)))

    @classmethod
    def zeros_like(cls, tensor):
        """Return a `Sparse` tensor with the same shape and data type as the given `tensor`."""

        return cls.zeros(tensor.shape)

    def as_dense(self):
        """Return a :class:`Dense` view of this `Sparse` tensor."""

        return self._get("dense", rtype=Dense)

    def div(self, other):
        return Sparse(Div(self, other))

    def elements(self, bounds=None):
        """
        Return a :class:`Stream` of this tensor's `(coord, number)` coordinate-value elements.

        `coord` is a :class:`Tuple` of :class:`U64` coordinates, and `number` is the element at `coord`.
        """

        bounds = handle_bounds(bounds)
        return self._get("elements", bounds, Stream)

    def mul(self, other):
        return Sparse(Mul(self, other))


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
    return rtype(form=Tile(tensor, multiples))


def where(cond, x, y):
    """
    Return a view of `x` and `y` depending on whether the corresponding element of `cond` is `True`.

    `cond`, `x`, and `y` must support broadcasting to the same shape.
    """

    return (cond.cast(Bool) * x) + (cond.logical_not() * y)


class Concatenate(Operator):
    def __init__(self, tensors, axis=None):
        if not hasattr(tensors, "__len__"):
            logging.debug(f"Concatenate({tensors}) will not support automatic differentiation")

        if axis:
            for tensor in tensors:
                if not is_literal(tensor.shape[axis]):
                    logging.debug(f"tensor {tensor} to concatenate noes not have a constant shape at axis {axis}")

        Operator.__init__(self, tensors, axis)

    def __repr__(self):
        return f"concat({self.subject}, {self.args})"

    @property
    def shape(self):
        if not hasattr(self.subject, "__len__"):
            raise ValueError(f"the concatenation of {self.subject} does not have a constant shape")

        return Shape.concatenate([t.shape for t in self.subject], self.args)

    def forward(self):
        params = {"tensors": self.subject}
        if self.args:
            params["axis"] = self.args

        # TODO: support concatenating Sparse tensors
        return Dense(form=ref.Post(uri(Dense) + "/concatenate", params))

    def backward(self, variable=None):
        if not isinstance(deref(self.subject), (list, tuple)):
            raise ValueError(f"the derivative of a tensor concatenation requires a constant list, not {self.subject}")

        # TODO: support concatenating Sparse tensors
        return Dense(Concatenate([derivative_of(t, variable) for t in self.subject], self.args))

    def gradients(self, loss):
        if not isinstance(deref(self.subject), (list, tuple)):
            raise ValueError(f"the gradients of a tensor concatenation requires a constant list, not {self.subject}")

        grads = Gradients()

        if isinstance(loss, Number):
            for tensor in self.subject:
                if operator(tensor):
                    grads.update(operator(tensor).gradients(loss))
                else:
                    grads[ref.hex_id(tensor)] = loss

            return grads

        axis = self.args if self.args else 0
        if axis is None:
            num_or_size_slices = len(self.subject)
        else:
            num_or_size_slices = [t.shape[axis] for t in self.subject]

        if not is_literal(num_or_size_slices):
            raise TypeError(f"the gradients of a concatenation require constant-shaped inputs, not {self.subject}")

        losses = loss.split(num_or_size_slices, axis)
        assert len(losses) == len(self.subject)

        for (tensor, loss) in zip(self.subject, losses):
            if operator(tensor):
                grads.update(operator(tensor).gradients(loss))
            else:
                grads[ref.hex_id(tensor)] = loss

        return grads


class Copy(Unary):
    def __repr__(self):
        return f"copy({self.subject})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return ref.Post(uri(Tensor) + "/copy_from", {"tensor": self.subject})

    def backward(self, variable=None):
        return derivative_of(self.subject, variable).copy()

    def gradients(self, loss):
        if operator(self.subject):
            return operator(self.subject).gradients(loss)

        return Gradients()


class Norm(Operator):
    def __repr__(self):
        if self.args:
            return f"norm({self.subject}[{self.args}])"
        else:
            return f"norm({self.subject})"

    @property
    def shape(self):
        if self.args is None:
            return self.subject.shape[:-2]
        else:
            return self.subject.shape.reduce(self.args)

    def forward(self):
        return NDArray.norm(self.subject, self.args)

    def backward(self, variable=None):
        return self.subject / self.subject.norm(self.args)

    def gradients(self, loss):
        loss *= self.backward()

        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients(loss))
        else:
            grads[ref.hex_id(self.subject)] = loss

        return grads


class Reduce(Operator):
    def __init__(self, tensor, axis=None, keepdims=False):
        Operator.__init__(self, tensor, _reduce_args(axis, keepdims))

    @property
    def shape(self):
        return Shape.reduce(self.subject.shape, **self.args)


class Sum(Reduce):
    def __repr__(self):
        if "axis" in self.args:
            return f"sum({self.subject}[{self.args['axis']}])"
        else:
            return f"sum({self.subject})"

    def forward(self):
        return NDArray.sum(self.subject, **self.args)

    def backward(self, variable=None):
        return derivative_of(self.subject).sum(**self.args)

    def gradients(self, loss):
        if "axis" not in self.args:
            loss = self.backward() * loss
        elif isinstance(loss, NDArray):
            # TODO: is this correct?
            loss = Dense.ones_like(self.subject) * loss

        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients(loss))
        else:
            grads[ref.hex_id(self.subject)] = loss

        return grads


class Transform(Operator):
    def backward(self, variable=None):
        rtype = type(self.subject) if isinstance(self.subject, Tensor) else Tensor
        d = type(self)(derivative_of(self.subject, variable), self.args)
        return rtype(form=d)

    def invert(self, loss):
        raise NotImplementedError(f"{self.__class__.__name__}.invert")

    def gradients(self, loss):
        if isinstance(loss, NDArray):
            loss = self.invert(loss)

        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients(loss))
        else:
            grads[ref.hex_id(self.subject)] = loss

        return grads


class Broadcast(Transform):
    def __repr__(self):
        return str(self.subject)

    def forward(self):
        return NDArray.broadcast(self.subject, self.args)


class Expand(Transform):
    def __repr__(self):
        return f"{self.subject}.expand({self.args})"

    @property
    def shape(self):
        return Shape.expand(self.subject.shape, self.args)

    def forward(self):
        return NDArray.expand_dims(self.subject, self.args)

    def invert(self, loss):
        return loss.reshape(self.subject.shape)


class Flip(Transform):
    def __repr__(self):
        return f"{self.subject}.flip({self.args})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return NDArray.flip(self.subject, self.args)

    def invert(self, loss):
        return loss.flip(self.args)


class Tile(Transform):
    def __init__(self, tensor, multiples):
        if not is_literal(multiples):
            raise ValueError(f"Tensor.tile requires a constant value for multiples, not {multiples}")

        Transform.__init__(self, tensor, multiples)

    def __repr__(self):
        return f"{self.subject}.tile({self.args})"

    @property
    def shape(self):
        return self.subject.shape.tile(self.args)

    def forward(self):
        return ref.Post(uri(Tensor) + "/tile", {"tensor": self.subject, "multiples": self.args})

    def invert(self, loss):
        if not isinstance(loss, NDArray):
            return loss

        if not is_literal(self.subject.shape):
            raise RuntimeError(f"inversion with respect to a tiled tensor requires a constant shape, not {self.shape}")

        dims = deref(self.subject.shape)
        multiples = ([1] * (len(self.args) - 1)) + [self.args] if isinstance(self.args, int) else self.args
        assert len(dims) == len(multiples)
        assert not any(m <= 0 for m in multiples)

        if all(m == 1 for m in multiples):
            return loss

        tiled_axes = [x for x, m in enumerate(multiples) if m != 1]
        if len(tiled_axes) == 1:
            [axis] = tiled_axes
            return loss.sum(axis, keepdims=True)

        sum_over = []
        for offsets in itertools.product(range(0, m) for m in multiples):
            bounds = [slice(offset, offset + dim) for offset, dim in zip(dims, offsets)]
            sum_over.append(loss[bounds])

        return sum(sum_over)


class Transpose(Transform):
    def __init__(self, subject, permutation=None):
        Transform.__init__(self, subject, permutation)

    def __repr__(self):
        if self.args:
            return f"{self.subject}.transpose({self.args})"
        else:
            return f"{self.subject}.T"

    @property
    def shape(self):
        return Shape.transpose(self.subject.shape, self.args)

    def forward(self):
        return NDArray.transpose(self.subject, self.args)

    def invert(self, loss):
        if self.args is None:
            inverse_permutation = None
        else:
            inverse_permutation = tuple(i for _x, i in sorted((x, i) for i, x in enumerate(self.args)))

        return loss.transpose(inverse_permutation)


class Reshape(Transform):
    def __repr__(self):
        return f"{self.subject}.reshape({self.args})"

    @property
    def shape(self):
        return Shape.reshape(self.subject.shape, self.args)

    def forward(self):
        return NDArray.reshape(self.subject, self.args)

    def invert(self, loss):
        return loss.reshape(self.subject.shape)


class Slice(Transform):
    def __init__(self, tensor, bounds):
        if not is_literal(tensor.shape):
            logging.debug(f"slice of {tensor} will not support automatic differentiation")

        Transform.__init__(self, tensor, bounds)

    def __repr__(self):
        return f"{self.subject}.slice({self.args})"

    @property
    def shape(self):
        return Shape.slice(self.subject.shape, self.args)

    def forward(self):
        return NDArray.slice(self.subject, self.args)

    def invert(self, loss):
        grad = Dense.zeros_like(self.subject)  # TODO: this should support Sparse tensors as well

        # TODO: there must be a better way to do this
        class SliceGradient(Operator):
            def __init__(self, grad):
                Operator.__init__(self, grad, None)

            def __repr__(self):
                return f"{self.subject}[{self.args}]"

            def __ns__(self, context):
                return deanonymize(self.subject, context)

            @property
            def shape(self):
                return self.subject.shape

            def forward(self):
                return self.subject

            def backward(self, _variable=None):
                return self.subject

        return Dense(SliceGradient(Dense(ref.After(grad[self.args].write(loss), ref.MethodSubject(grad)))))


def _reduce_args(axis=None, keepdims=False):
    args = {}

    if axis is not None:
        args["axis"] = axis

    if keepdims:
        args["keepdims"] = keepdims

    return args
