"""TinyChain `State` s, like `Map`, `Tuple`, and `Op`."""

import inspect

from tinychain.util import *


class State(object):
    """
    A TinyChain state, such as a `Chain` or `Op` or `Value`.

    Do not subclass `State` directly. Use a more specific type instead.
    """

    __uri__ = URI("/state")

    def __init__(self, form):
        from tinychain import reflect

        self.__form__ = form

        if isinstance(form, URI):
            self.__uri__ = form
        elif reflect.is_ref(form):
            self.__uri__ = uri(form)

        reflect.meta.gen_headers(self)

    def __dbg__(self):
        return [form_of(self)]

    def __eq__(self, _other):
        raise NotImplementedError("State does not support equality; use a more specific type")

    def __json__(self):
        from tinychain import reflect

        form = form_of(self)

        if reflect.is_ref(form):
            return to_json(form)
        else:
            return {str(uri(self)): [to_json(form)]}

    def __id__(self):
        return hex_id(form_of(self))

    def __ns__(self, cxt):
        deanonymize(form_of(self), cxt)

        if isinstance(self.__form__, URI):
            self.__uri__ = self.__form__

    def __ref__(self, name):
        if hasattr(form_of(self), "__ref__"):
            return self.__class__(get_ref(form_of(self), name))
        else:
            return self.__class__(URI(name))

    def __repr__(self):
        return f"{self.__class__.__name__}({form_of(self)})"

    def _get(self, name, key=None, rtype=None):
        from tinychain.new_state.ref import MethodSubject, Get

        subject = MethodSubject(self, name)
        op_ref = Get(subject, key)
        rtype = State if rtype is None or not issubclass(rtype, State) else rtype
        return rtype(op_ref)

    def _put(self, name, key=None, value=None):
        from tinychain.new_state.ref import MethodSubject, Put
        from tinychain.new_state.value import Nil

        subject = MethodSubject(self, name)
        return Nil(Put(subject, key, value))

    def _post(self, name, params, rtype):
        from tinychain.new_state.ref import MethodSubject, Post

        subject = MethodSubject(self, name)
        op_ref = Post(subject, params)
        rtype = State if rtype is None or not issubclass(rtype, State) else rtype
        return rtype(op_ref)

    def _delete(self, name, key=None):
        from tinychain.new_state.ref import MethodSubject, Delete
        from tinychain.new_state.value import Nil

        subject = MethodSubject(self, name)
        return Nil(Delete(subject, key))

    def cast(self, dtype):
        """Attempt to cast this `State` into the given `dtype`."""

        # TODO: allow casting to a type known only at run-time
        if not inspect.isclass(dtype) or not issubclass(dtype, State):
            raise NotImplementedError("dtype to cast into must be known at compile-time")

        from tinychain.new_state.ref import Get
        return dtype(Get(dtype, self))

    def copy(self):
        """Create a new `State` by copying this one."""

        return self._get("copy", rtype=self.__class__)

    def dtype(self):
        """Return the native base :class:`Class` of this `State`."""
        return self._get("class", rtype=Class)

    def hash(self):
        """Return the SHA256 hash of this `State` as an :class:`Id`."""

        from tinychain.new_state.value import Id
        return self._get("hash", rtype=Id)

    def is_none(self):
        """Return `Bool(true)` if this `State` is :class:`Nil`."""

        from tinychain.new_state.value import Bool
        return self._get("is_none", rtype=Bool)

    def is_some(self):
        """
        Return `Bool(true)` if this `State` is not :class:`Nil`.

        This is defined as `self.is_none().logical_not()`.
        """

        return self.is_none().logical_not()
