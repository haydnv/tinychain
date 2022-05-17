"""A discrete :class:`State`."""

import inspect

from .base import _Base
from .interface import Functional
from .scalar.ref import form_of, get_ref, hex_id, is_ref, Ref
from .uri import uri, URI
from .context import deanonymize, to_json


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
            while isinstance(form, State):
                form = form_of(form)

            # make sure to capture the URI of the given form
            if isinstance(form, URI):
                self.__uri__ = form
            elif is_ref(form) and hasattr(form, "__uri__"):
                assert uri(form)
                self.__uri__ = uri(form)

            self.__form__ = form

        assert not isinstance(form_of(self), State)

        _Base.__init__(self)

    def __json__(self):
        form = form_of(self)

        if uri(form) == uri(self):
            return to_json(form)
        else:
            return {str(uri(self)): [to_json(form)]}

    def __id__(self):
        return hex_id(form_of(self))

    def __ns__(self, cxt, name_hint):
        form = form_of(self)

        deanonymize(form, cxt, name_hint)

        if isinstance(self.__form__, URI):
            self.__uri__ = self.__form__

    def __ref__(self, name):
        if hasattr(form_of(self), "__ref__"):
            return self.__class__(form=get_ref(form_of(self), name))
        else:
            return self.__class__(form=StateRef(self, name))

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
        return dtype(form=Get(dtype, self))

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

class Stream(State, Functional):
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

    def first(self):
        """Return the first item in this `Stream`, or `Nil` if the `Stream` is empty."""

        return self._get("first", rtype=State)

    def flatten(self):
        """Flatten a `Stream` of `Stream` s into a single `Stream` of their component elements."""

        return self._get("flatten", rtype=Stream)


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


class StateRef(Ref):
    def __init__(self, state, name):
        self.state = state
        self.__uri__ = URI(name)

    def __repr__(self):
        is_auto_assigned = False

        address = str(uri(self)).split('_')[-1]
        try:
            is_auto_assigned = int(address, 16)
        except ValueError:
            pass

        if is_auto_assigned:
            return repr(self.state)
        else:
            return str(uri(self))

    def __id__(self):
        return hex_id(self.state)

    def __json__(self):
        return to_json(uri(self))

    def __ns__(self, cxt, name_hint):
        deanonymize(self.state, cxt, name_hint + '_' + str(uri(self))[1:].replace('/', '_'))
