"""User-defined ops"""

import dataclasses
import typing

from ..generic import resolve_class
from ..state import State
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


Args = typing.TypeVar("Args")
RType = typing.TypeVar("RType", bound=State)


class Post(Op, typing.Generic[Args, RType]):
    """A function which can be called via a POST request."""

    __uri__ = URI(Op) + "/post"

    def __init__(self, form):
        assert str(form) != "$loss"
        Op.__init__(self, form)

    def __call__(self, *args, **kwargs):
        if hasattr(self, "__orig_class__"):
            sig, rtype = typing.get_args(self.__orig_class__)
            assert dataclasses.is_dataclass(sig)
            sig = dataclasses.fields(sig)
            rtype = resolve_class(rtype)

            if len(args) > len(sig):
                raise TypeError(f"{self} takes {len(sig)} arguments but got {args}")

            params = {}
            for i, field in enumerate(sig):
                if i < len(args):
                    params[field.name] = args[i]
                elif field.name in kwargs:
                    params[field.name] = kwargs[field.name]
                elif field.default is not dataclasses.MISSING:
                    params[field.name] = field.default
                else:
                    raise TypeError(f"missing value for parameter {field.name} in call to {self} with arguments {args}")
        else:
            assert not args
            params = kwargs

        return rtype(form=ref.Post(self, params))


class Delete(Op):
    """A function which can be called via a DELETE request."""

    __uri__ = URI(Op) + "/delete"

    def __call__(self, key=None):
        return ref.Delete(self, key)
