import typing

from .generic import Tuple
from .scalar.number import U64
from .scalar.ref import is_literal


class Shape(Tuple):
    __spec__ = typing.Tuple[U64, ...]

    def ndim(self, require_constant=False):
        if require_constant and not hasattr(self, "__len__"):
            raise RuntimeError("this operation requires a constant number of dimensions")

        return self.len()

    def broadcast(self, other):
        ndim = max(self.ndim(True), other.ndim(True))
        assert ndim > 0

        shape = [1] * ndim

        left = [1] * (ndim - self.ndim()) + list(self)
        right = [1] * (ndim - other.ndim()) + list(other)
        assert len(left) == len(right)

        for x, (l, r) in reversed(enumerate(zip(left, right))):
            if is_literal(l) and is_literal(r):
                if l == r:
                    dim = l
                elif l == 1:
                    dim = r
                elif r == 1:
                    dim = l
                else:
                    raise ValueError(f"cannot broadcast dimensions {l} and {r}")

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
                raise ValueError(f"broadcast requires at least one of the dimensions {l} and {r} to be constant")

            shape[x] = dim

        return Shape(shape)

    def reduce(self, axes):
        if not is_literal(axes):
            raise ValueError(f"Shape.reduce requires constant axes, not {axes}")

        if not isinstance(axes, (list, tuple)):
            axes = tuple(axes)

        ndim = self.ndim(True)
        axes = tuple(ndim + x if x < 0 else x for x in axes)
        return Shape(tuple(dim for x, dim in enumerate(self) if x not in axes))
