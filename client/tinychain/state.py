"""A discrete :class:`State`."""

import inspect
import typing

from .reflect import is_ref, MethodStub
from .util import deanonymize, form_of, get_ref, hex_id, to_json, uri, URI


class _Base(object):
    def __init__(self):
        # TODO: is there a better place for this?
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue

            if isinstance(attr, MethodStub):
                method = attr.method(self, name)
                setattr(self, name, method)

    def _get(self, name, key=None, rtype=None):
        from .scalar.ref import Get, MethodSubject

        subject = MethodSubject(self, name)
        op_ref = Get(subject, key)
        rtype = _resolve_rtype(rtype)
        return rtype(form=op_ref)

    def _put(self, name, key=None, value=None):
        from .scalar.ref import MethodSubject, Put
        from .scalar.value import Nil

        subject = MethodSubject(self, name)
        return Nil(Put(subject, key, value))

    def _post(self, name, params, rtype):
        from .scalar.ref import MethodSubject, Post

        subject = MethodSubject(self, name)
        op_ref = Post(subject, params)
        rtype = _resolve_rtype(rtype)
        return rtype(form=op_ref)

    def _delete(self, name, key=None):
        from .scalar.ref import Delete, MethodSubject
        from .scalar.value import Nil

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

    def __init__(self, form=None):
        if form is None:
            if not hasattr(self, "__form__") or self.__form__ is None:
                raise ValueError(f"instance of {self.__class__.__name__} has no form")
        else:
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

        from .scalar.ref import Get
        return dtype(Get(dtype, self))

    def copy(self):
        """Create a new `State` by copying this one."""

        return self._get("copy", rtype=self.__class__)

    def dtype(self):
        """Return the native base :class:`Class` of this `State`."""
        return self._get("class", rtype=Class)

    def hash(self):
        """Return the SHA256 hash of this `State` as an :class:`Id`."""

        from .scalar.value import Id

        return self._get("hash", rtype=Id)

    def is_none(self):
        """Return `Bool(true)` if this `State` is :class:`Nil`."""

        from .scalar.number import Bool
        return self._get("is_none", rtype=Bool)

    def is_some(self):
        """
        Return `Bool(true)` if this `State` is not :class:`Nil`.

        This is defined as `self.is_none().logical_not()`.
        """

        return self.is_none().logical_not()


# A stream of `State` s

class Stream(State):
    """A stream of states which supports functional methods like `fold` and `map`."""

    __uri__ = uri(State) + "/stream"

    @classmethod
    def range(cls, range):
        """
        Construct a new :class:`Stream` of :class:`Number` s in the given `range`.

        `range` can be a positive :class:`Number`, `(start, stop)`, or `(start, stop, step)`
        """

        from .scalar.ref import Get
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


class Object(State):
    """A user-defined type"""

    __uri__ = uri(State) + "/object"


class Class(Object):
    """A TinyChain class (possibly a user-defined class)."""

    __uri__ = uri(Object) + "/class"

    def __call__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Class.__call__ accepts args or kwargs but not both")

        from .scalar.ref import Get, MethodSubject

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


def _resolve_rtype(rtype, default=State):
    if typing.get_origin(rtype) is tuple:
        from .generic import Tuple
        return Tuple.expect(rtype)
    elif typing.get_origin(rtype) is dict:
        from .generic import Map
        return Map.expect(rtype)
    elif inspect.isclass(rtype) and issubclass(rtype, State):
        return rtype
    else:
        return default
