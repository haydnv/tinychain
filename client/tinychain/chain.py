"""Data structures responsible for keeping track of mutations to a :class:`Value` or :class:`Collection`"""

from .scalar import ref
from .scalar.value import Nil
from .state import State
from .uri import uri, URI


class Chain(State):
    """Data structure responsible for keeping track of mutations to a :class:`Value` or :class:`Collection`."""

    __uri__ = uri(State) + "/chain"

    def __new__(cls, form):
        if ref.is_ref(form):
            return State.__new__(cls)

        elif isinstance(form, State):

            class _Chain(cls, type(form)):
                def __json__(self):
                    return cls.__json__(self)

            return State.__new__(_Chain)

        else:
            raise ValueError(f"Chain subject must be a State, not {form}")

    def __init__(self, form):
        # in this case, the type of form is significant
        # so we can't call State.__init__ because it will discard the type of the form

        if isinstance(form, URI):
            self.__uri__ = form
        elif ref.is_ref(form) and hasattr(form, "__uri__"):
            self.__uri__ = uri(form)

        self.__form__ = form

    def set(self, value):
        """Update the value of this `Chain`."""

        return ref.Put(uri(self), None, value)

    # TODO: delete these overrides and make MethodSubject compatible with Chain
    def _get(self, name, key=None, rtype=State):
        op_ref = ref.Get(uri(self).append(name), key)
        rtype = Nil if rtype is None else rtype
        return rtype(form=op_ref)

    def _put(self, name, key=None, value=None):
        return Nil(ref.Put(uri(self).append(name), key, value))

    def _post(self, name, params, rtype=State):
        op_ref = ref.Post(uri(self).append(name), params)
        rtype = Nil if rtype is None else rtype
        return rtype(form=op_ref)

    def _delete(self, name, key=None):
        return Nil(ref.Delete(uri(self).append(name), key))


class Block(Chain):
    """A :class:`Chain` which keeps track of the entire update history of its subject."""

    __uri__ = uri(Chain) + "/block"


class Sync(Chain):
    """
    A :class:`Chain` which keeps track of only the current transaction's operations,
    in order to recover from a transaction failure (e.g. if the host crashes).
    """

    __uri__ = uri(Chain) + "/sync"
