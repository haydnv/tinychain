import inspect

from .math.equation import NativeFunction, NativeStateFunction
from .reflect.stub import ReflectionStub, StateFunctionStub
from .reflect import method, op
from .scalar.ref import With


def closure(*deps):
    """
    Annotation to capture data referenced by an :class:`Op` and return a `Closure`.

    The returned `Closure` can be called with the same parameters as the :class:`Op`.

    **Important**: the captured data may be large (e.g. an entire `Table`) if a closure is serialized over the network.

    Example:
        .. highlight:: python
        .. code-block:: python

            @tc.post
            def outer(x: Number, y: Number):
                @tc.closure(x, y)
                @tc.get
                def inner(z: tc.Number):
                    return x * y * z

                return inner
    """

    return lambda op: With(deps, op)


def get(form):
    """Annotation for a callable function or method specifying that it is a GET :class:`Op`."""

    return StateFunctionStub(method.Get, form) if _is_method(form) else op.Get(form)


def put(form):
    """Annotation for a callable function or method specifying that it is a PUT :class:`Op`."""

    return StateFunctionStub(method.Put, form) if _is_method(form) else op.Put(form)


def post(form):
    """Annotation for a callable function or method specifying that it is a POST :class:`Op`."""

    return StateFunctionStub(method.Post, form) if _is_method(form) else op.Post(form)


def delete(form):
    """Annotation for a callable function or method specifying that it is a DELETE :class:`Op`."""

    return StateFunctionStub(method.Delete, form) if _is_method(form) else op.Delete(form)


def differentiable(form):
    """Annotation for a callable method specifying that it returns a type of differentiable :class:`Operator`."""

    return StateFunctionStub(NativeStateFunction, form) if _is_method(form) else NativeFunction(form)


def reflect(form):
    """Annotation for a method produced by reflection over its instance."""

    assert _is_method(form)
    return ReflectionStub(form)


def _is_method(form):
    if not callable(form):
        return False
    elif inspect.ismethod(form):
        return True

    param_names = list(inspect.signature(form).parameters.keys())
    if param_names and param_names[0] == "self":
        return True

    return False
