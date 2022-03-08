import typing

from ..decorators import post_op
from ..generic import Map
from ..scalar.number import Number


def product(tuple: typing.Tuple[Number]) -> Number:
    """Compute the product of a `Tuple` of `Number` s."""

    p = tuple.fold('n', Map(p=1), post_op(lambda n, p: Map(p=Number.mul(n, p))))['p']
    return Number(p)


def sum(tuple: typing.Tuple[Number]) -> Number:
    """Compute the sum of a `Tuple` of `Number` s."""

    s = tuple.fold('n', Map(s=0), post_op(lambda n, s: Map(s=Number.add(n, s))))['s']
    return Number(s)
