"""User-defined ops"""

from ..uri import URI

from . import ref
from .base import Scalar


class Op(Scalar):
    """A callable function."""

    __uri__ = URI(Scalar) + "/op"


class Get(Op):
    """A function which can be called via a GET request."""

    __uri__ = URI(Op) + "/get"

    def __call__(self, key=None):
        return ref.Get(self, key)


class Put(Op):
    """A function which can be called via a PUT request."""

    __uri__ = URI(Op) + "/put"

    def __call__(self, key=None, value=None):
        return ref.Put(self, key, value)


class Post(Op):
    """A function which can be called via a POST request."""

    __uri__ = URI(Op) + "/post"

    # TODO: this should retain the return type, in the case of a reflected op
    def __call__(self, params=None, **kwargs):
        if params is None:
            params = kwargs
        elif not isinstance(params, dict):
            raise TypeError(f"to call a POST op requires named parameters, not {params}")
        elif kwargs:
            raise ValueError("POST call takes a Map or kwargs, not both:", params, kwargs)

        params = params if params else kwargs
        return ref.Post(self, params)


class Delete(Op):
    """A function which can be called via a DELETE request."""

    __uri__ = URI(Op) + "/delete"

    def __call__(self, key=None):
        return ref.Delete(self, key)
