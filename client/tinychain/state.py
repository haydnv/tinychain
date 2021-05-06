"""Tinychain `State`\s, such as `Map`, `Tuple`, and `Op`."""

from . import reflect
from .ref import MethodSubject, Ref, OpRef
from .util import *


class State(object):
    """
    A Tinychain state, such as a `Chain` or `Op` or `Value`.

    Do not subclass `State` directly. Use a more specific type instead.
    """

    __uri__ = URI("/state")

    def __init__(self, form):
        self.__form__ = form

        if isinstance(form, URI):
            self.__uri__ = form

        reflect.gen_headers(self)

    def __json__(self):
        form = form_of(self)
        if isinstance(form, URI) or isinstance(form, OpRef):
            return to_json(form)
        else:
            return {str(uri(self)): [to_json(form)]}

    def __ns__(self, cxt):
        form = form_of(self)

        if isinstance(form, OpRef):
            deanonymize(form, cxt)

    def __ref__(self, name):
        return self.__class__(URI(name))

    def __repr__(self):
        return f"{self.__class__.__name__}({form_of(self)})"

    def _method(self, name):
        if isinstance(form_of(self), OpRef):
            return MethodSubject(self, name)

        subject = uri(self).append(name)
        if subject.startswith("/state") and subject.path() != uri(self.__class__):
            raise ValueError(
                f"cannot call instance method {name} with an absolute path: {subject}")

        return subject

    def _get(self, name, key=None, rtype=None):
        subject = self._method(name)
        rtype = State if rtype is None else rtype
        return rtype(OpRef.Get(subject, key))

    def _put(self, name, key=None, value=None):
        from .value import Nil

        subject = self._method(name)
        return Nil(OpRef.Put(subject, key, value))

    def _post(self, _method_name, rtype=None, **params):
        from .value import Nil

        subject = self._method(_method_name)
        rtype = State if rtype is None else rtype
        return rtype(OpRef.Post(subject, **params))

    def _delete(self, name, key=None):
        from .value import Nil

        subject = self._method(name)
        return Nil(OpRef.Delete(subject, key))

    def dtype(self):
        """Return the native :class:`Class` of this `State`."""
        return Class(OpRef.Get(uri(self) + "/class"))


class Map(State):
    """A key-value map whose keys are `Id`\s and whose values are `State`\s."""

    __uri__ = uri(State) + "/map"

    def __getitem__(self, key):
        return self._get("", key)

    def __json__(self):
        return to_json(form_of(self))


class Tuple(State):
    """A tuple of `State`\s."""

    __uri__ = uri(State) + "/tuple"

    def __json__(self):
        return to_json(form_of(self))

    def __getitem__(self, key):
        return self._get("", key)


# Scalar types

class Scalar(State):
    """
    An immutable `State` which always resides entirely in the host's memory.

    Do not subclass `Scalar` directly. Use :class:`Value` instead.
    """

    __uri__ = uri(State) + "/scalar"

    def __json__(self):
        return to_json(form_of(self))


# User-defined Ops

class Op(Scalar):
    """A callable function."""

    __uri__ = uri(Scalar) + "/op"


class GetOp(Op):
    """A function which can be called via a GET request."""

    __uri__ = uri(Op) + "/get"

    def __call__(self, key=None):
        return OpRef.Get(self, key)


class PutOp(Op):
    """A function which can be called via a PUT request."""

    __uri__ = uri(Op) + "/put"

    def __call__(self, key=None, value=None):
        return OpRef.Put(self, key, value)


class PostOp(Op):
    """A function which can be called via a POST request."""

    __uri__ = uri(Op) + "/post"

    def __call__(self, **params):
        return OpRef.Post(self, **params)


class DeleteOp(Op):
    """A function which can be called via a DELETE request."""

    __uri__ = uri(Op) + "/delete"

    def __call__(self, key=None):
        return OpRef.Delete(self, key)


Op.Get = GetOp
Op.Put = PutOp
Op.Post = PostOp
Op.Delete = DeleteOp


# User-defined object types

class Class(State):
    """A user-defined Tinychain class."""

    __uri__ = uri(State) + "/object/class"

    def __json__(self):
        return {str(uri(Class)): to_json(form_of(self))}


class Instance(State):
    """An instance of a user-defined :class:`Class`."""

    __uri__ = uri(State) + "/object/instance"

