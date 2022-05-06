"""An n-dimensional array of numbers."""

from ..decorators import post
from ..generic import Map, Tuple
from ..interface import Equality, Interface, Order
from ..math.operator import *
from ..scalar.bound import handle_bounds
from ..scalar.number import Bool, F32, F64, Number, UInt, U64
from ..scalar import ref
from ..state import Class, Stream
from ..util import form_of, hex_id, uri

from .base import Collection


class Shape(Tuple):
    __spec__ = typing.Tuple[U64, ...]

    def reduce_shape(self, axes):
        return ref.Post(uri(Tensor).append("reduce_shape"), {"shape": self, "axes": axes})


class NDArray(Interface):
    def slice(self, bounds):
        return ref.Get(ref.MethodSubject(self), bounds)

    def expand_dims(self, axis=-1):
        return ref.Get(ref.MethodSubject(self, "expand_dims"), axis)

    def flip(self, axis):
        return ref.Get(ref.MethodSubject(self, "flip"), axis)

    def norm(self):
        return ref.Get(ref.MethodSubject(self, "norm"))

    def reshape(self, shape):
        return ref.Get(ref.MethodSubject(self, "reshape"), shape)

    def transpose(self, permutation=None):
        return ref.Get(ref.MethodSubject(self, "transpose"), permutation)


class Tensor(Collection, Equality, Numeric, Order, Trigonometric, NDArray):
    """An n-dimensional array of numbers."""

    __uri__ = uri(Collection) + "/tensor"
    __spec__ = (Shape, Number)

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

        spec = (shape, dtype)

        if not isinstance(shape, Shape):
            shape = Shape(shape)

        class _Tensor(cls):
            __spec__ = spec

            @classmethod
            def create(cls):
                return cls(ref.Get(cls, (shape, dtype)))

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
    def load(cls, shape, data, dtype=F32):
        """
        Load a `Tensor` from an existing data set.

        Example:
            .. highlight:: python
            .. code-block:: python

                coords = [[0, 0, 1], [0, 1, 0]]
                values = [1, 2]
                sparse = tc.tensor.Sparse.load([2, 3, 4], zip(coords, values))
                dense = tc.tensor.Dense.load([2, 3, 4], values, tc.I32)
        """

        return cls.expect(shape, dtype)(ref.Get(uri(cls) + "/load", ((shape, dtype), data)))

    def __getitem__(self, bounds):
        return self.slice(bounds)

    def __setitem__(self, bounds, value):
        raise NotImplementedError("use Tensor.write instead")

    def __matmul__(self, other):
        return Tensor(MatMul(self, other))

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

        rtype = Tensor
        if isinstance(form_of(self.shape), (list, tuple)):
            if isinstance(form_of(axis), int):
                shape = list(form_of(self.shape))
                shape.insert(form_of(axis), 1)
                rtype = Tensor.expect(shape, self.dtype)

        return rtype(form=Expand(self, axis))

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

    def norm(self):
        """Compute the Frobenius norm (aka Euclidean norm) of a matrix or batch of matrices."""

        return Tensor(Norm(self))

    def pow(self, other):
        return Tensor(Pow(self, other))

    def product(self, axis=None):
        """Calculate the product of this `Tensor` along the given `axis`, or the total product if no axis is given."""

        rtype = Number if axis is None else Tensor
        return self._get("product", axis, rtype)

    def reshape(self, shape):
        """Return a view of this `Tensor` with the given `shape`."""

        return Tensor.expect(shape, self.dtype)(form=Reshape(self, shape))

    @property
    def shape(self):
        """Return the shape of this `Tensor`."""

        return self._get("shape", rtype=Shape)

    @property
    def size(self):
        """Return the size of this `Tensor` (the product of its `shape`)."""

        return self._get("size", rtype=UInt)

    def slice(self, bounds):
        parent = self
        bounds = handle_bounds(bounds)

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

        num_or_size_splits = form_of(num_or_size_splits)
        if not ref.is_literal(num_or_size_splits):
            raise ValueError(f"Tensor.split requires a constant num_or_size_splits, not {num_or_size_splits}")

        if not ref.is_literal(axis):
            raise ValueError(f"Tensor.split requires a constant axis, not {axis}")

        dim = form_of(self.shape[axis])
        if not int(dim) == dim:
            raise RuntimeError(f"Tensor.split requires a constant dimension to split")

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

    def sum(self, axis=None):
        """Calculate the sum of this `Tensor` along the given `axis`, or the total sum if no axis is given."""

        rtype = Number if axis is None else Tensor
        return self._get("sum", axis, rtype)

    def transpose(self, permutation=None):
        """
        Return a view of this `Tensor` with its axes transposed according to the given permutation.

        If no permutation is given, the axes will be inverted (e.g. `(0, 1, 2)` inverts to `(2, 1, 0)`).
        """

        dtype = self.dtype
        shape = None

        if hasattr(self.shape, "__len__"):
            if permutation is None:
                shape = reversed(self.shape)
            elif hasattr(permutation, "__iter__"):
                shape = [self.shape[x] for x in permutation]

        if shape is None:
            rtype = Tensor
        else:
            rtype = Tensor.expect(shape, dtype)

        return rtype(form=Transpose(self, permutation))

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

        return Dense(form=Concatenate(tensors, axis))

    @classmethod
    def constant(cls, shape, value):
        """Return a `Dense` tensor filled with the given `value`."""

        dtype = type(value) if isinstance(value, Number) else Number
        return cls.expect(shape, dtype)(ref.Get(uri(cls) + "/constant", (shape, value)))

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
    def zeros(cls, shape, dtype=F32):
        """
        Return a `Sparse` tensor with the given shape and data type.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.expect(shape, dtype)(ref.Get(cls, (shape, dtype)))

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
    return rtype(form=ref.Post(uri(Tensor) + "/tile", {"tensor": tensor, "multiples": multiples}))


def where(cond, x, y):
    """
    Return a view of `x` and `y` depending on whether the corresponding element of `cond` is `True`.

    `cond`, `x`, and `y` must support broadcasting to the same shape.
    """

    return (cond.cast(Bool) * x) + (cond.logical_not() * y)


class Concatenate(Dual):
    def forward(self):
        params = {"tensors": self.subject}
        if self.args:
            params["axis"] = self.args

        # TODO: support concatenating Sparse tensors
        return Dense(form=ref.Post(uri(Dense) + "/concatenate", params))

    def backward(self, variable=None):
        return Concatenate([derivative_of(t, variable) for t in self.subject], self.args)

    def gradients(self, loss):
        from ..ml import Variable  # TODO: remove this import

        grads = Gradients()

        axis = self.args if self.args else 0
        if axis is None:
            num_or_size_slices = len(self.subject)
        else:
            num_or_size_slices = [t.shape[axis] for t in self.subject]

        losses = loss.split(num_or_size_slices, axis)
        assert len(losses) == len(self.subject)

        for (tensor, loss) in zip(self.subject, losses):
            if operator(tensor):
                grads.update(operator(tensor).gradients(loss))
            elif isinstance(tensor, Variable):
                grads[hex_id(tensor)] = loss

        return grads


class Copy(Unary):
    def forward(self):
        return ref.Post(uri(Tensor) + "/copy_from", {"tensor": self.subject})

    def gradients(self, loss):
        if operator(self.subject):
            return form_of(self.subject).gradients(loss)

        return Gradients()


class Norm(Unary):
    def forward(self):
        return NDArray.norm(self.subject)

    def backward(self, variable=None):
        raise NotImplementedError("derivative of norm")

    def gradients(self, loss):
        raise NotImplementedError("gradients of norm")


class Transform(Operator):
    def backward(self, variable=None):
        rtype = type(self.subject) if isinstance(self.subject, Tensor) else Tensor
        d = type(self)(derivative_of(self.subject, variable), self.args)
        return rtype(form=d)

    def gradients(self, loss):
        if not operator(self.subject):
            logging.info(f"{self.subject} is assumed to be constant and has no gradient")
            return Gradients()

        return operator(self.subject).gradients(loss)


class Expand(Transform):
    def forward(self):
        return NDArray.expand_dims(self.subject, self.args)

    def gradients(self, loss):
        loss = loss if isinstance(loss, Number) else loss.reshape(self.subject.shape)
        return Transform.gradients(self, loss)


class Flip(Transform):
    def forward(self):
        return NDArray.flip(self.subject, self.args)

    def gradients(self, loss):
        loss = loss if isinstance(loss, Number) else loss.flip(self.args)
        return Transform.gradients(self, loss)


class Transpose(Transform):
    def __init__(self, subject, permutation=None):
        Transform.__init__(self, subject, permutation)

        if permutation is None:
            self.inverse_permutation = None
        else:
            self.inverse_permutation = tuple(i for _x, i in sorted((x, i) for i, x in enumerate(permutation)))

    def forward(self):
        return NDArray.transpose(self.subject, self.args)

    def gradients(self, loss):
        loss = loss if isinstance(loss, Number) else loss.transpose(self.inverse_permutation)
        return Transform.gradients(self, loss)


class Reshape(Transform):
    def forward(self):
        return NDArray.reshape(self.subject, self.args)

    def gradients(self, loss):
        loss = loss if isinstance(loss, Number) else loss.reshape(self.subject.shape)
        return Transform.gradients(self, loss)


class Slice(Transform):
    def forward(self):
        return NDArray.slice(self.subject, self.args)

    def gradients(self, loss):
        if isinstance(loss, Number):
            return Transform.gradients(loss)

        grad = Dense.zeros_like(self.subject)  # TODO: this should support Sparse tensors as well
        grad = Dense(ref.After(grad[self.args].write(loss), ref.MethodSubject(grad, "")))
        return Transform.gradients(self, grad)
