from ...scalar.ref import deref, is_literal

from .base import Tensor
from .operator import Tile


def split(tensor, num_or_size_splits, axis=0):
    """
    Split the given `tensor` into multiple slices along the given `axis`.

    This method requires a constant `num_or_size_splits`, `axis`, and `self.shape[axis]`.

    If `num_or_size_splits` is a `Number`, the `tensor` will be sliced along `axis` `num_or_size_splits` times;
    if `self.shape[axis] % num_or_size_splits != 0` then a `ValueError` error will be raised.

    If `num_or_size_splits` is a `Tuple` with length `n` then the `tensor` will be split into `n` slices
    each with `shape[axis] == num_or_size_splits[axis]`; if the sum of `num_or_size_splits` is not equal to
    `self.shape[axis]` then a `ValueError` error will be raised.
    """

    num_or_size_splits = deref(num_or_size_splits)
    if not is_literal(num_or_size_splits):
        raise ValueError(f"split requires a constant num_or_size_splits, not {num_or_size_splits}")

    if not is_literal(axis):
        raise ValueError(f"split requires a constant axis, not {axis}")

    if is_literal(tensor.shape[axis]):
        dim = deref(tensor.shape[axis])
    else:
        raise RuntimeError(f"to split {tensor} requires a constant dimension to split, not {tensor.shape[axis]}")

    if isinstance(num_or_size_splits, (list, tuple)):
        if sum([deref(dim) for dim in num_or_size_splits]) != dim:
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
        slices.append(tensor[bounds])

    return slices


def tile(tensor, multiples):
    """Construct a new `Tensor` by tiling the given `tensor` `multiples` times.

    The values of `tensor` are repeated `multiples[x]` times along the `x`th axis of the output.
    `multiples` must be a positive integer or a `Tuple` of length `tensor.ndim`.
    """

    return Tensor(form=Tile(tensor, multiples))
