"""A discrete :class:`State`."""

import inspect
import logging

from ..util import deanonymize, form_of, get_ref, hex_id, to_json, uri, URI


class _Base(object):
    def __init__(self):
        from ..reflect import MethodStub

        # TODO: is there a better place for this?
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue

            if isinstance(attr, MethodStub):
                setattr(self, name, attr.method(self, name))

    def _get(self, name, key=None, rtype=None):
        from .ref import MethodSubject, Get

        subject = MethodSubject(self, name)
        op_ref = Get(subject, key)
        rtype = State if rtype is None or not issubclass(rtype, State) else rtype
        return rtype(form=op_ref)

    def _put(self, name, key=None, value=None):
        from .ref import MethodSubject, Put
        from .value import Nil

        subject = MethodSubject(self, name)
        return Nil(Put(subject, key, value))

    def _post(self, name, params, rtype):
        from .ref import MethodSubject, Post

        subject = MethodSubject(self, name)
        op_ref = Post(subject, params)
        rtype = State if rtype is None or not issubclass(rtype, State) else rtype
        return rtype(form=op_ref)

    def _delete(self, name, key=None):
        from .ref import MethodSubject, Delete
        from .value import Nil

        subject = MethodSubject(self, name)
        return Nil(Delete(subject, key))


class Interface(_Base):
    """The base class of a client-defined `Interface`"""


class State(_Base):
    """
    A TinyChain state, such as a `Chain` or `Op` or `Value`.

    Do not subclass `State` directly. Use a more specific type instead.
    """

    __uri__ = URI("/state")

    def __init__(self, form):
        from ..reflect import is_ref
        from .ref import Ref

        self.__form__ = form

        if isinstance(form, URI):
            self.__uri__ = form
        elif is_ref(form) and hasattr(form, "__uri__"):
            self.__uri__ = uri(form)

        _Base.__init__(self)

    def __dbg__(self):
        return [form_of(self)]

    def __eq__(self, _other):
        raise NotImplementedError("State does not support equality; use a more specific type")

    def __json__(self):
        from ..reflect import is_ref
        from .ref import MethodSubject, Ref

        form = form_of(self)

        if is_ref(form):
            return to_json(form)
        else:
            return {str(uri(self)): [to_json(form)]}

    def __id__(self):
        return hex_id(form_of(self))

    def __ns__(self, cxt):
        form = form_of(self)

        deanonymize(form, cxt)

        if isinstance(self.__form__, URI):
            self.__uri__ = self.__form__

    def __ref__(self, name):
        if hasattr(form_of(self), "__ref__"):
            return self.__class__(form=get_ref(form_of(self), name))
        else:
            return self.__class__(form=URI(name))

    def __repr__(self):
        if hasattr(self, "__form__") and self.__form__:
            return f"{self.__class__.__name__}({form_of(self)})"
        else:
            return f"instance of {self.__class__.__name__}"

    def cast(self, dtype):
        """Attempt to cast this `State` into the given `dtype`."""

        # TODO: allow casting to a type known only at run-time
        if not inspect.isclass(dtype) or not issubclass(dtype, State):
            raise NotImplementedError("dtype to cast into must be known at compile-time")

        from .ref import Get
        return dtype(Get(dtype, self))

    def copy(self):
        """Create a new `State` by copying this one."""

        return self._get("copy", rtype=self.__class__)

    def dtype(self):
        """Return the native base :class:`Class` of this `State`."""
        return self._get("class", rtype=Class)

    def hash(self):
        """Return the SHA256 hash of this `State` as an :class:`Id`."""

        from .value import Id
        return self._get("hash", rtype=Id)

    def is_none(self):
        """Return `Bool(true)` if this `State` is :class:`Nil`."""

        from .value import Bool
        return self._get("is_none", rtype=Bool)

    def is_some(self):
        """
        Return `Bool(true)` if this `State` is not :class:`Nil`.

        This is defined as `self.is_none().logical_not()`.
        """

        return self.is_none().logical_not()


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

    @classmethod
    def range(cls, range):
        """Return a stream of numbers in the given `range`.

        `range` can be a positive number, a 2-Tuple like `(start, stop)`, or a 3-Tuple like `(start, stop, step)`
        """

        from .ref import Get
        return cls(Get(uri(cls) + "/range", range))

    def aggregate(self):
        return self._get("aggregate", rtype=Stream)

    def filter(self, op):
        """Return a new `Stream` containing only those elements of this `Stream` where the given `op` returns `True`."""

        return self._post("filter", {"op": op}, Stream)

    def first(self):
        """Return the first item in this `Stream`, or `Nil` if the `Stream` is empty."""

        return self._get("first", rtype=State)

    def flatten(self):
        """Flatten a `Stream` of `Stream` s into a single `Stream` of their component elements."""

        return self._get("flatten", rtype=Stream)

    def for_each(self, op):
        """Run the given `op` for each item in this `Stream`, then return the last result.

        This is useful when you need to execute an `op` for its side-effects and not its return value.
        """

        rtype = op.rtype if hasattr(op, "rtype") else State
        return self._post("for_each", {"op": op}, rtype)

    def fold(self, item_name, initial_state, op):
        """Run the given `op` for each item in this `Stream` along with the previous result.

        `op` must be a POST Op. The stream item to handle will be passed with the given `item_name` as its name.
        """

        rtype = type(initial_state) if isinstance(initial_state, State) else State
        return self._post("fold", {"item_name": item_name, "value": initial_state, "op": op}, rtype)

    def map(self, op):
        """Return a new `Stream` whose items are the results of running `op` on each item of this `Stream`."""

        return self._post("map", {"op": op}, Stream)



# User-defined object types

class Object(State):
    """A user-defined type"""

    __uri__ = uri(State) + "/object"


class Class(Object):
    """A TinyChain class (possibly a user-defined class)."""

    __uri__ = uri(Object) + "/class"

    def __call__(self, *args, **kwargs):
        from .ref import MethodSubject, Get

        if args and kwargs:
            raise ValueError("Class.__call__ accepts args or kwargs but not both")

        subject = MethodSubject(self)
        if args:
            return Get(subject, args)
        else:
            return Get(subject, kwargs)


class Instance(Object):
    """An instance of a user-defined :class:`Class`."""

    __uri__ = uri(Object) + "/instance"

    def copy(self):
        raise NotImplementedError("abstract method")
