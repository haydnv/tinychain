from tinychain.reflect.meta import MethodStub
from tinychain.reflect import method, op


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
