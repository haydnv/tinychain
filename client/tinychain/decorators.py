from tinychain.ref import With
from tinychain.reflect.meta import MethodStub
from tinychain.reflect import method, op
from tinychain.util import requires


def closure(op):
    """
    Annotation to capture data referenced by an :class:`Op` and return a `Closure`.

    The returned `Closure` can be called with the same parameters as the :class:`Op`.

    **Important**: the captured data may be large (e.g. an entire `Table`) if a closure is serialized over the network.
    For fine-grained control over what data is captured, use the :class:`With` flow control.
    """

    return With(requires(op), op)


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
