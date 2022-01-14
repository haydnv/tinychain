from tinychain.error import TinyChainError
from tinychain.ref import After, If, With
from tinychain.reflect.meta import MethodStub
from tinychain.reflect import method, op
from tinychain.state import State


def after(when, then):
    """
    Delay resolving `then` until `when` has been resolved.

    Use this in situations where `then` depends on a side-effect of `when`, e.g. `after(table.update(), table.count())`.
    """
    
    rtype = type(then) if isinstance(then, State) else State
    return rtype(After(when, then))


def cond(if_condition, then, or_else):
    """
    Branch execution conditionally (i.e. an if statement).

    `if_condition` must resolve to a `Bool`. If true, `then` will be returned, otherwise `or_else`.
    """

    then_is_error = isinstance(then, TinyChainError)
    or_else_is_error = isinstance(or_else, TinyChainError)

    rtype = State
    if not then_is_error and or_else_is_error:
        rtype = type(then) if isinstance(then, State) else State
    elif then_is_error and not or_else_is_error:
        rtype = type(or_else) if isinstance(or_else, State) else State
    elif isinstance(then, State) and isinstance(or_else, State):
        if isinstance(type(or_else), type(then)):
            rtype = type(then)
        elif isinstance(type(then), type(or_else)):
            rtype = type(or_else)

    return rtype(If(if_condition, then, or_else))


def closure(*deps):
    """
    Annotation to capture data referenced by an :class:`Op` and return a `Closure`.

    The returned `Closure` can be called with the same parameters as the :class:`Op`.

    **Important**: the captured data may be large (e.g. an entire `Table`) if a closure is serialized over the network.

    Example:
        .. highlight:: python
        .. code-block:: python

            @tc.post_op
            def outer(x: Number, y: Number):
                @tc.closure(x, y)
                @tc.get_op
                def inner(z: tc.Number):
                    return x * y * z

                return inner
    """

    return lambda op: With(deps, op)


def get_method(form):
    """Annotation for a callable method specifying that it is a GET method."""
    return MethodStub(method.Get, form)


def put_method(form):
    """Annotation for a callable method specifying that it is a PUT method."""
    return MethodStub(method.Put, form)


def post_method(form):
    """Annotation for a callable method specifying that it is a POST method."""
    return MethodStub(method.Post, form)


def delete_method(form):
    """Annotation for a callable method specifying that it is a DELETE method."""
    return MethodStub(method.Delete, form)


def get_op(form):
    """Annotation for a callable function specifying that it is a GET :class:`Op`."""
    return op.Get(form)


def put_op(form):
    """Annotation for a callable function specifying that it is a PUT :class:`Op`."""
    return op.Put(form)


def post_op(form):
    """Annotation for a callable function specifying that it is a POST :class:`Op`."""
    return op.Post(form)


def delete_op(form):
    """Annotation for a callable function specifying that it is a DELETE :class:`Op`."""
    return op.Delete(form)
