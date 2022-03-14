import inspect

from .reflect.meta import MethodStub
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


def hidden(form):
    """Annotation for a method which can only be called at compile-time (i.e. is hidden from run-time callers)"""

    form.hidden = True
    return form


def get(form):
    """Annotation for a callable function or method specifying that it is a GET :class:`Op`."""

    if _is_method(form):
        return MethodStub(method.Get, form)
    else:
        return op.Get(form)


def put(form):
    """Annotation for a callable function or method specifying that it is a PUT :class:`Op`."""

    if _is_method(form):
        return MethodStub(method.Put, form)
    else:
        return op.Put(form)


def post(form):
    """Annotation for a callable function or method specifying that it is a POST :class:`Op`."""

    if _is_method(form):
        return MethodStub(method.Post, form)
    else:
        return op.Post(form)


def delete(form):
    """Annotation for a callable function or method specifying that it is a DELETE :class:`Op`."""

    if _is_method(form):
        return MethodStub(method.Delete, form)
    else:
        return op.Delete(form)


def operator(form):
    """Annotation for a differentiable operator"""

    if _is_method(form):
        return MethodStub(method.Operator, form)
    else:
        return op.Operator(form)


def _is_method(form):
    if inspect.ismethod(form):
        return True

    param_names = list(inspect.signature(form).parameters.keys())
    if param_names and param_names[0] == "self":
        return True

    return False
