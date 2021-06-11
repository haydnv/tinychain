"""Data structures responsible for keeping track of mutations to a :class:`Value` or :class:`Collection`"""

from tinychain import ref
from tinychain.reflect.meta import Meta
from tinychain.state import State
from tinychain.util import uri, URI


class Chain(State, metaclass=Meta):
    """Data structure responsible for keeping track of mutations to a :class:`Value` or :class:`Collection`."""

    __uri__ = uri(State) + "/chain"

    def __new__(cls, spec):
        if isinstance(spec, ref.Ref) or isinstance(spec, URI):
            return State.__new__(cls)

        elif isinstance(spec, State):

            class _Chain(cls, type(spec)):
                def __json__(self):
                    return cls.__json__(self)

            return State.__new__(_Chain)

        else:
            raise ValueError(f"Chain subject must be a State, not {spec}")

    def set(self, value):
        """Update the value of this `Chain`."""

        return ref.Put(uri(self), None, value)


class Block(Chain):
    """A :class:`Chain` which keeps track of the entire update history of its subject."""

    __uri__ = uri(Chain) + "/block"


class Sync(Chain):
    """
    A :class:`Chain` which keeps track of only the current transaction's operations,
    in order to recover from a transaction failure (e.g. if the host crashes).
    """

    __uri__ = uri(Chain) + "/sync"
