"""User-defined ops"""

from tinychain import ref
from tinychain.state import Scalar
from tinychain.util import uri


class Op(Scalar):
    """A callable function."""

    __uri__ = uri(Scalar) + "/op"


class Get(Op):
    """A function which can be called via a GET request."""

    __uri__ = uri(Op) + "/get"

    def __call__(self, key=None):
        return ref.Get(self, key)


class Put(Op):
    """A function which can be called via a PUT request."""

    __uri__ = uri(Op) + "/put"

    def __call__(self, key=None, value=None):
        return ref.Put(self, key, value)


class Post(Op):
    """A function which can be called via a POST request."""

    __uri__ = uri(Op) + "/post"

    def __call__(self, params=None, **kwargs):
        if kwargs and params is not None:
            raise ValueError("Post takes a Map or kwargs, not both:", params, kwargs)

        params = params if params else kwargs
        return ref.Post(self, params)


class Delete(Op):
    """A function which can be called via a DELETE request."""

    __uri__ = uri(Op) + "/delete"

    def __call__(self, key=None):
        return ref.Delete(self, key)
