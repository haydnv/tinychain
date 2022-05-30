import functools
import operator
import typing

from .generic import Tuple
from .scalar.bound import Range
from .scalar.number import Number, U64
from .scalar.ref import deref, is_literal, get_ref
from .state import State
from .uri import uri


class Shape(Tuple):
    __spec__ = typing.Tuple[U64, ...]

    def __new__(cls, form):
        if hasattr(form, "__iter__"):
            spec = tuple(U64 for _ in form)
            return State.__new__(cls.expect(spec))
        else:
            return State.__new__(cls)

    def __add__(self, other):
        if hasattr(self, "__len__") and hasattr(other, "__len__"):
            return Shape(tuple(deref(self)) + tuple(deref(other)))
        else:
            raise ValueError(f"can only append Shapes with a literal number of dimensions, not {self} and {other}")

    def __radd__(self, other):
        return Shape(other) + self

    def __getitem__(self, x):
        if x is None:
            raise ValueError(f"invalid axis: {x}")

        spec = typing.get_args(self.__spec__) if typing.get_args(self.__spec__) else self.__spec__

        if isinstance(x, Range):
            x = x.to_slice()

        if isinstance(x, slice):
            start = deref(x.start)
            stop = deref(x.stop)

            if x.step is not None:
                raise NotImplementedError(f"slice with step: {x}")

            if len(spec) != 2 or (len(spec) == 2 and spec[1] is not Ellipsis):
                # the contents may be literal, so compute the slice now if possible
                if hasattr(deref(self), "__getitem__"):
                    if is_literal(start) and is_literal(stop):
                        start = _index_of(start, len(self), 0)
                        stop = _index_of(stop, len(self), len(self))
                        return Shape([self[i] for i in range(start, stop)])

            return self._get("", Range.from_slice(x), Shape)

        if is_literal(x):
            x = deref(x)
            if hasattr(deref(self), "__getitem__"):
                try:
                    item = deref(self)[x]
                except IndexError as e:
                    raise IndexError(f"{self} has no axis {x}: {e}")

                if uri(self) == uri(self.__class__):
                    return item
                else:
                    return get_ref(item, uri(self).append(x))

        return self._get("", x, U64)

    def __repr__(self):
        if hasattr(self, "__len__"):
            return f"[{', '.join(str(dim) if is_literal(dim) else 'U64' for dim in self)}]"
        else:
            return "[...]"

    @classmethod
    def concatenate(cls, shapes, axis=0):
        if not shapes:
            raise ValueError("cannot concatenate an empty list of shapes")
        elif not hasattr(shapes, "__len__"):
            raise ValueError(f"can only concatenate a literal list of shapes, not {shapes}")
        else:
            shapes = [Shape(shape) for shape in shapes]

        ndim = shapes[0].ndim(True, "concatenate")
        for shape in shapes:
            if shape.ndim(True, "concatenate") != ndim:
                raise ValueError("shapes to concatenate must have the same number of dimensions")

        if isinstance(deref(axis), int):
            axis = deref(axis)
            axis = ndim + axis if axis < 0 else axis
        else:
            raise ValueError(f"Shape.concatenate requires a literal axis, not {axis}")

        dim = 0
        for shape in shapes:
            if shape[axis] is None:
                raise ValueError(f"dimension for concatenation at axis {axis} is unknown: {shape[axis]}")

            if is_literal(shape[axis]):
                dim += deref(shape[axis])
            else:
                raise ValueError(f"Shape.concatenate requires literal dimensions along the axis {axis}")

        concatenated = [None] * ndim
        concatenated[axis] = dim

        for x in (x for x in range(ndim) if x != axis):
            for shape in shapes:
                if concatenated[x] is None:
                    concatenated[x] = shape[x]
                elif concatenated[x] != shape[x]:
                    raise ValueError(f"cannot concatenate {shapes} due to inconsistent dimension at axis {x}")

            if concatenated[x] is None:
                raise ValueError(f"shape of concatenated tensor is not kown at axis {x}")

        return Shape(concatenated)

    def ndim(self, require_literal=False, op_name="this operation"):
        if require_literal and not hasattr(self, "__len__"):
            raise RuntimeError(f"to {op_name} {self} requires a literal number of dimensions")

        if hasattr(self, "__len__"):
            return len(self)
        else:
            return self.len()

    def broadcast(self, other):
        ndim = max(self.ndim(True, "broadcast"), other.ndim(True, "broadcast"))

        assert int(ndim) == ndim

        if not ndim:
            return Shape(tuple())

        shape = [1] * ndim

        left = [1] * (ndim - self.ndim()) + list(self)
        right = [1] * (ndim - other.ndim()) + list(other)
        assert len(left) == len(right)

        for x, (l, r) in reversed(list(enumerate(zip(left, right)))):
            if is_literal(l) and is_literal(r):
                if l == r:
                    dim = l
                elif l == 1:
                    dim = r
                elif r == 1:
                    dim = l
                else:
                    raise ValueError(f"cannot broadcast dimensions {l} and {r} (shapes are {left} and {right})")

            elif is_literal(l):
                if l == 1:
                    dim = r
                else:
                    dim = l  # assume l == r

            elif is_literal(r):
                if r == 1:
                    dim = l
                else:
                    dim = r  # assume l == r

            else:
                raise ValueError(f"broadcast requires at least one of the dimensions {l} and {r} to be literal")

            shape[x] = dim

        return Shape(shape)

    def expand(self, axis=None):
        if not hasattr(self, "__len__"):
            raise RuntimeError(f"Shape.expand requires a literal number of dimensions")
        elif len(self) == 0:
            raise RuntimeError(f"cannot expand shape {self}")

        if not is_literal(axis):
            raise ValueError(f"Shape.expand requires a literal axis, not {axis}")

        if axis is None:
            return Shape(self + [1])
        else:
            axis = len(self) + axis if axis < 0 else axis
            assert axis >= 0
            return Shape(self[:axis] + [1] + self[axis:])

    def reduce(self, axis=None, keepdims=False):
        if not is_literal(axis):
            raise ValueError(f"Shape.reduce requires a literal axis, not {axis}")

        if not is_literal(keepdims):
            return ValueError(f"the keepdims parameter of Shape.reduce must be a literal, not {keepdims}")

        if not keepdims and axis is None:
            return Shape(tuple())

        ndim = self.ndim(True, "reduce")
        axis = ndim + axis if axis < 0 else axis
        if keepdims:
            return Shape(tuple(1 if x == axis else dim for x, dim in enumerate(self)))
        else:
            return Shape(tuple(dim for x, dim in enumerate(self) if x != axis))

    def reshape(self, new_shape):
        if is_literal(new_shape):
            new_shape = deref(new_shape)
        else:
            return new_shape

        for (x, dim) in enumerate(new_shape):
            if dim is not None and dim < 0:
                raise ValueError(f"invalid dimension for reshape at axis {x}: {dim}")

        if len([dim for dim in new_shape if dim is None]) > 1:
            raise ValueError(f"Shape.reshape supports a maximum of one unknown dimension, not {new_shape}")

        if is_literal(self):
            this_size = int(functools.reduce(operator.mul, deref(self)))
            for x in range(len(new_shape)):
                if new_shape[x] is None:
                    that_size = int(functools.reduce(operator.mul, new_shape[:x] + new_shape[x + 1:]))
                    if this_size % that_size == 0:
                        new_shape[x] = this_size // that_size
                    else:
                        raise ValueError(f"cannot reshape {self} into {new_shape}")

            that_size = int(functools.reduce(operator.mul, (dim for dim in new_shape if dim is not None)))
            if this_size != that_size:
                raise ValueError(f"cannot reshape {self} into {new_shape}")

            return Shape(new_shape)
        elif any(dim is None for dim in new_shape):
            raise ValueError(f"{self} does not support reshape with an unknown dimension: {new_shape}")
        else:
            return Shape(new_shape)

    def slice(self, bounds):
        if not hasattr(bounds, "__iter__"):
            raise ValueError(f"the shape of a Tensor slice requires literal-length bounds, not {bounds}")

        if hasattr(self, "__len__") and len(bounds) > len(self):
            raise ValueError(f"{bounds} are out of bounds for shape {self}")

        shape = []
        for x, bound in enumerate(bounds):
            if isinstance(bound, Range):
                bound = bound.to_slice()

            if isinstance(bound, slice):
                start = 0 if bound.start is None else deref(bound.start)
                stop = deref(self[x]) if bound.stop is None else deref(bound.stop)
                if not is_literal((start, stop)):
                    raise ValueError(f"the shape of a Tensor slice requires a literal bound, not {(start, stop)}")

                if start < 0 or stop < 0:
                    if is_literal(self[x]):
                        dim = self[x]
                    else:
                        raise RuntimeError(f"Shape.slice requires a literal dimension for axis {x}, not {self[x]}")

                    start = dim + start if start < 0 else start
                    stop = dim + stop if stop < 0 else stop

                shape.append(stop - start)
            elif isinstance(bound, (int, Number)):
                # no-op -- dimension elided
                pass
            else:
                raise ValueError(f"invalid axis bound: {bound}")

        for x in range(len(bounds), self.ndim(True, "slice")):
            shape.append(self[x])

        return Shape(shape)

    def tile(self, multiples):
        if not is_literal(multiples):
            raise ValueError(f"Shape.tile requires a literal number or tuple for multiples, not {multiples}")

        multiples = deref(multiples)
        if isinstance(multiples, (list, tuple)):
            if not is_literal(self):
                raise ValueError(f"only a literal Shape supports tiling {multiples} times per-axis")

            assert len(self) == len(multiples)
            return Shape([dim * m for dim, m in zip(self, multiples)])
        elif isinstance(multiples, int):
            return Shape(self[:-1] + [self[-1] * multiples])
        else:
            raise ValueError(f"invalid number of multiples for Shape.tile: {multiples}")

    def transpose(self, permutation=None):
        if permutation is None:
            if not hasattr(self, "__reversed__"):
                raise RuntimeError(f"Shape.transpose requires a literal-length shape, not {self}")

            return Shape(tuple(reversed(self)))
        elif is_literal(permutation):
            return Shape(tuple(self[x] for x in permutation))
        else:
            raise ValueError(f"Shape.transpose requires a literal permutation, not {permutation}")


def _index_of(i, length, default):
    if i is None:
        idx = default
    elif i < 0:
        idx = length + i
    else:
        idx = i

    return idx
