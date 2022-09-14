import itertools
import logging

from ...math.operator import derivative_of, gradients, Gradients, Operator, Unary
from ...scalar.number import Number
from ...scalar.ref import deref, is_literal, same_as, After, Post
from ...shape import Shape
from ...uri import URI


class Cast(Operator):
    def __init__(self, source, dtype):
        Operator.__init__(self, source, dtype)

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        from .base import NDArray
        return NDArray.cast(self.subject, self.args)

    def backward(self, variable=None):
        return Cast(derivative_of(self.subject, variable), self.args)

    def gradients(self, loss):
        return gradients(self.subject, loss)


# TODO: support concatenating Sparse tensors
class Concatenate(Operator):
    def __init__(self, tensors, axis=None):
        if not hasattr(tensors, "__len__"):
            logging.debug(f"Concatenate({tensors}) will not support automatic differentiation")

        if axis and is_literal(axis):
            for tensor in tensors:
                if not is_literal(tensor.shape[deref(axis)]):
                    logging.debug(f"tensor {tensor} to concatenate noes not have a literal shape at axis {axis}")

        Operator.__init__(self, tensors, axis)

    def __repr__(self):
        return f"concat({self.subject}, {self.args})"

    @property
    def shape(self):
        if not hasattr(self.subject, "__len__"):
            raise ValueError(f"concatenation does not have a literal shape")

        return Shape.concatenate([t.shape for t in self.subject], self.args)

    def forward(self):
        from .base import Dense

        params = {"tensors": self.subject}
        if self.args:
            params["axis"] = self.args

        return Dense(form=Post(URI(Dense) + "/concatenate", params))

    def backward(self, variable=None):
        if not isinstance(deref(self.subject), (list, tuple)):
            raise ValueError(f"the derivative of a tensor concatenation requires a literal list")

        from .base import Dense, NDArray

        d = [derivative_of(t, variable) for t in self.subject]

        if all(same_as(d_i, 0) for d_i in d):
            return 0
        elif all(same_as(d_i, 1) for d_i in d):
            return 1
        elif all(isinstance(d_i, NDArray) for d_i in d):
            return Dense(Concatenate(d, self.args))
        else:
            d = [derivative_of(t, variable, keepdims=True) for t in self.subject]
            assert all(isinstance(d_i, NDArray) for d_i in d)
            return Dense(Concatenate(d, self.args))

    def gradients(self, loss):
        if not isinstance(deref(self.subject), (list, tuple)):
            raise ValueError(f"the gradients of a tensor concatenation requires a literal list")

        grads = Gradients()

        if isinstance(loss, Number):
            for tensor in self.subject:
                grads += gradients(tensor, loss)

            return grads

        axis = self.args if self.args else 0
        if axis is None:
            num_or_size_slices = len(self.subject)
        else:
            num_or_size_slices = [t.shape[axis] for t in self.subject]

        if not is_literal(num_or_size_slices):
            raise TypeError(f"the gradients of a concatenation require literal-shaped inputs, not {self.subject}")

        from .functions import split

        losses = split(loss, num_or_size_slices, axis)
        assert len(losses) == len(self.subject)

        for (tensor, loss) in zip(self.subject, losses):
            grads += gradients(tensor, loss)

        return grads


class Copy(Unary):
    def __repr__(self):
        return f"copy({self.subject})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        from .base import Tensor
        return Post(URI(Tensor, "copy_from"), {"tensor": self.subject})

    def backward(self, variable=None):
        from .base import NDArray

        d = derivative_of(self.subject, variable)
        if isinstance(d, NDArray):
            return d.copy()
        else:
            return d

    def gradients(self, loss):
        return gradients(self.subject, loss)


class Read(Operator):
    def __init__(self, tensor, bounds):
        Operator.__init__(self, tensor, bounds)

    def __repr__(self):
        return f"{self.subject}[{self.args}]"

    @property
    def shape(self):
        return Shape(tuple())

    def forward(self):
        from .base import NDArray
        return NDArray.slice(self.subject, self.args)

    def backward(self, variable=None):
        return derivative_of(self.subject).slice(self.args)


class Reduce(Operator):
    def __init__(self, tensor, axis=None, keepdims=False):
        Operator.__init__(self, tensor, _reduce_args(axis, keepdims))

    def __args__(self):
        return self.subject, self.args.get("axis"), self.args.get("keepdims")

    @property
    def shape(self):
        return Shape.reduce(self.subject.shape, **self.args)

    def gradients(self, loss):
        if not is_literal(self.subject.ndim):
            raise RuntimeError(f"gradients of Reduce require a literal number of dimensions, not {self.subject.ndim}")

        # here we explicitly backpropagate the loss to the subject of this op
        # so we know it will be multiplied by its derivative there; here we just need to set the correct shape

        if self.args.get("axis") is None:
            from .base import Dense
            loss = loss * Dense.ones([1] * deref(self.subject.ndim))
            return gradients(self.subject, loss)
        elif not self.args.get("keepdims"):
            return gradients(self.subject, loss.expand_dims(self.args["axis"]))
        else:
            return gradients(self.subject, loss)


class Norm(Operator):
    def __init__(self, tensor, axis=None, keepdims=False):
        Operator.__init__(self, tensor, _reduce_args(axis, keepdims))

    def __repr__(self):
        if self.args:
            return f"norm({self.subject}[{self.args}])"
        else:
            return f"norm({self.subject})"

    def __args__(self):
        return self.subject, self.args.get("axis"), self.args.get("keepdims")

    @property
    def shape(self):
        if self.args is None:
            return self.subject.shape[:-2]
        else:
            return self.subject.shape.reduce(**self.args)

    def forward(self):
        from .base import NDArray
        return NDArray.norm(self.subject, **self.args)

    def backward(self, variable=None):
        return self.subject / self.subject.norm(**self.args)

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Max(Reduce):
    def __repr__(self):
        if "axis" in self.args:
            return f"max({self.subject}[{self.args['axis']}])"
        else:
            return f"max({self.subject})"

    def forward(self):
        from .base import NDArray
        return NDArray.max(self.subject, **self.args)

    def backward(self, variable=None):
        from .base import NDArray

        subject = derivative_of(self.subject, variable)

        if isinstance(subject, NDArray):
            return subject.max(**self.args)
        else:
            return subject


class Min(Reduce):
    def __repr__(self):
        if "axis" in self.args:
            return f"min({self.subject}[{self.args['axis']}])"
        else:
            return f"min({self.subject})"

    def forward(self):
        from .base import NDArray
        return NDArray.min(self.subject, **self.args)

    def backward(self, variable=None):
        from .base import NDArray

        subject = derivative_of(self.subject, variable)

        if isinstance(subject, NDArray):
            return subject.min(**self.args)
        else:
            return subject


class Product(Reduce):
    def __repr__(self):
        if "axis" in self.args:
            return f"product({self.subject}[{self.args['axis']}])"
        else:
            return f"product({self.subject})"

    def forward(self):
        from .base import NDArray
        return NDArray.product(self.subject, **self.args)

    def backward(self, variable=None):
        from .base import NDArray

        subject = derivative_of(self.subject, variable)

        if isinstance(subject, NDArray):
            return subject.product(**self.args)
        else:
            return subject


class Sum(Reduce):
    def __repr__(self):
        if "axis" in self.args:
            return f"sum({self.subject}[{self.args['axis']}])"
        else:
            return f"sum({self.subject})"

    def forward(self):
        from .base import NDArray
        return NDArray.sum(self.subject, **self.args)

    def backward(self, variable=None):
        from .base import NDArray

        subject = derivative_of(self.subject, variable)

        if isinstance(subject, NDArray):
            return subject.sum(**self.args)
        else:
            return subject


class Transform(Operator):
    def __init__(self, subject, args):
        from .base import NDArray

        if not isinstance(subject, NDArray):
            raise TypeError(f"transform requires an instance of NDArray, not {subject}")

        Operator.__init__(self, subject, args)

    def backward(self, variable=None):
        from .base import NDArray, Tensor
        d = derivative_of(self.subject, variable)

        if isinstance(d, NDArray):
            return Tensor(type(self)(d, self.args))
        else:
            return d

    def invert(self, loss):
        raise NotImplementedError(f"{self.__class__.__name__}.invert")

    def gradients(self, loss):
        from .base import NDArray

        if isinstance(loss, NDArray):
            loss = self.invert(loss)

        return gradients(self.subject, loss)


class Broadcast(Transform):
    def __repr__(self):
        return str(self.subject)

    @property
    def shape(self):
        return self.subject.shape.broadcast(self.args)

    def forward(self):
        from .base import NDArray
        return NDArray.broadcast(self.subject, self.args)


class Expand(Transform):
    def __repr__(self):
        return f"{self.subject}.expand({self.args})"

    @property
    def shape(self):
        return Shape.expand(self.subject.shape, self.args)

    def forward(self):
        from .base import NDArray
        return NDArray.expand_dims(self.subject, self.args)

    def invert(self, loss):
        return loss.sum(self.args)


class Flip(Transform):
    def __repr__(self):
        return f"{self.subject}.flip({self.args})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        from .base import NDArray
        return NDArray.flip(self.subject, self.args)

    def invert(self, loss):
        return loss.flip(self.args)


class Reshape(Transform):
    def __repr__(self):
        return f"{self.subject}.reshape({self.args})"

    @property
    def shape(self):
        return Shape.reshape(self.subject.shape, self.args)

    def forward(self):
        from .base import NDArray
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
        from .base import NDArray
        return NDArray.slice(self.subject, self.args)

    def invert(self, loss):
        if not is_literal(self.subject.shape):
            raise RuntimeError(f"the gradients of a Slice require a literal source shape, not {self.subject.shape}")

        from .base import Dense

        grad = Dense.zeros_like(self.subject)  # TODO: this should support Sparse tensors as well

        # TODO: there must be a better way to do this
        class SliceGradient(Operator):
            def __init__(self, grad):
                Operator.__init__(self, grad, None)

            def __args__(self):
                return self.subject,

            def __repr__(self):
                return f"{self.subject}[{self.args}]"

            def __ns__(self, cxt, name_hint):
                return cxt.deanonymize(self.subject, name_hint + "_subject")

            @property
            def shape(self):
                return grad.shape

            def forward(self):
                return self.subject

            def backward(self, _variable=None):
                return self.subject

        return Dense(SliceGradient(Dense(After(grad[self.args].write(loss), URI(grad)))))


class Tile(Transform):
    def __init__(self, tensor, multiples):
        if not is_literal(multiples):
            raise ValueError(f"Tensor.tile requires a literal value for multiples, not {multiples}")

        Transform.__init__(self, tensor, multiples)

    def __repr__(self):
        return f"{self.subject}.tile({self.args})"

    @property
    def shape(self):
        return self.subject.shape.tile(self.args)

    def forward(self):
        from .base import Tensor
        return Post(URI(Tensor) + "/tile", {"tensor": self.subject, "multiples": self.args})

    def invert(self, loss):
        from .base import NDArray

        if not isinstance(loss, NDArray):
            return loss

        if not is_literal(self.subject.shape):
            raise RuntimeError(f"inversion with respect to a tiled tensor requires a literal shape, not {self.shape}")

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
        from .base import NDArray
        return NDArray.transpose(self.subject, self.args)

    def invert(self, loss):
        if self.args is None:
            inverse_permutation = None
        else:
            inverse_permutation = tuple(i for _x, i in sorted((x, i) for i, x in enumerate(self.args)))

        return loss.transpose(inverse_permutation)


def _reduce_args(axis=None, keepdims=False):
    args = {}

    if axis is not None:
        args["axis"] = axis

    if keepdims:
        args["keepdims"] = keepdims

    return args
