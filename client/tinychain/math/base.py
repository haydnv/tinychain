from ..state import State

from .interface import Numeric


# TODO: add a generic type to `Functional` and return an instance of that type
def product(functional) -> Numeric:
    """Compute the product of a :class:`Functional` of :class:`Numeric` s."""

    from ..decorators import post_op
    from ..generic import Map

    p = functional.fold('n', Map(p=1), post_op(lambda n, p: Map(p=Numeric.mul(n, p))))['p']
    return type("Product", (State, Numeric), {})(p)


# TODO: add a generic type to `Functional` and return an instance of that type
def sum(functional) -> Numeric:
    """Compute the sum of a :class:`Functional` of :class:`Numeric` s."""

    from ..decorators import post_op
    from ..generic import Map

    s = functional.fold('n', Map(s=0), post_op(lambda n, s: Map(s=Numeric.add(n, s))))['s']
    return type("Sum", (State, Numeric), {})(s)
