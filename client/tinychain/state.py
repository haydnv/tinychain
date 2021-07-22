"""Tinychain `State` s, such as `Map`, `Tuple`, and `Op`."""
import inspect

from tinychain import ref, reflect
from tinychain.util import *


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

        reflect.meta.gen_headers(self)

    def __json__(self):
        form = form_of(self)
        if isinstance(form, URI) or isinstance(form, ref.Op):
            return to_json(form)
        else:
            return {str(uri(self)): [to_json(form)]}

    def __ns__(self, cxt):
        form = form_of(self)

        if isinstance(form, ref.Op):
            deanonymize(form, cxt)

    def __ref__(self, name):
        return self.__class__(URI(name))

    def __repr__(self):
        return f"{self.__class__.__name__}({form_of(self)})"

    def _method(self, name):
        if isinstance(form_of(self), ref.Op):
            return ref.MethodSubject(self, name)

        subject = uri(self).append(name)
        if subject.startswith("/state") and subject.path() != uri(self.__class__):
            raise ValueError(
                f"cannot call instance method {name} with an absolute path: {subject}")

        return subject

    def _get(self, name, key=None, rtype=None):
        subject = self._method(name)
        rtype = State if rtype is None else rtype
        return rtype(ref.Get(subject, key))

    def _put(self, name, key=None, value=None):
        from .value import Nil

        subject = self._method(name)
        return Nil(ref.Put(subject, key, value))

    def _post(self, _method_name, params, rtype):
        from .value import Nil

        subject = self._method(_method_name)
        if rtype is None:
            return Nil(ref.Post(subject.params))
        else:
            return rtype(ref.Post(subject, params))

    def _delete(self, name, key=None):
        from .value import Nil

        subject = self._method(name)
        return Nil(ref.Delete(subject, key))

    def dtype(self):
        """Return the native :class:`Class` of this `State`."""
        return Class(ref.Get(uri(self) + "/class"))


class Map(State):
    """A key-value map whose keys are `Id`s and whose values are `State` s."""

    __uri__ = uri(State) + "/map"

    def __init__(self, *args, **kwargs):
        if args:
            if kwargs:
                raise ValueError(f"Map accepts a form or kwargs, not both (got {kwargs})")

            [form] = args
            State.__init__(self, form)
        else:
            State.__init__(self, kwargs)

    def __getitem__(self, key):
        return self._get("", key)

    def __json__(self):
        return to_json(form_of(self))


class Tuple(State):
    """A tuple of `State` s."""

    __uri__ = uri(State) + "/tuple"

    def __json__(self):
        return to_json(form_of(self))

    def __getitem__(self, key):
        return self._get("", key)

    def map(self, op):
        rtype = op.rtype if hasattr(op, "rtype") else State
        return self._post("map", Map(op=op), rtype)


# Scalar types

class Scalar(State):
    """
    An immutable `State` which always resides entirely in the host's memory.

    Do not subclass `Scalar` directly. Use :class:`Value` instead.
    """

    __uri__ = uri(State) + "/scalar"

    def __json__(self):
        return to_json(form_of(self))


# A stream of `State` s

class Stream(State):
    """A stream of states which supports functional methods like `fold` and `map`."""

    __uri__ = uri(State) + "/stream"

    def aggregate(self, initial_value, op):
        return self._post("aggregate", Map(acc=initial_value, op=op), type(initial_value))

    def for_each(self, op):
        rtype = op.rtype if hasattr(op, "rtype") else State
        return self._post("for_each", Map(op=op), rtype)

    def fold(self, initial_value, op):
        return self._post("fold", Map(acc=initial_value, op=op), type(initial_value))

    def map(self, op):
        rtype = op.rtype if hasattr(op, "rtype") else State
        return self._post("fold", Map(op=op), rtype)


# User-defined object types

class Class(State):
    """A user-defined Tinychain class."""

    __uri__ = uri(State) + "/object/class"

    def __json__(self):
        return {str(uri(Class)): to_json(form_of(self))}


class Instance(State):
    """An instance of a user-defined :class:`Class`."""

    __uri__ = uri(State) + "/object/instance"

