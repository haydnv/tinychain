"""An n-dimensional array with both :class:`Dense` and :class:`Sparse` implementations"""

from .base import NDArray, Tensor, Dense, Sparse
from .einsum import einsum
from .functions import split, tile
from .operator import Copy, Transform
