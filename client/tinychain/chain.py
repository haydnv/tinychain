from .ref import OpRef
from .state import Scalar, State
from .util import *


class Chain(State):
    """
    Data structure responsible for keeping track of mutations to a :class:`Value`.
    """

    __uri__ = uri(State) + "/chain"

    def set(self, value):
        """Update the value of this `Chain`."""

        return OpRef.Put(uri(self).append("subject"), None, value)

    def subject(self, key=None):
        """Return a reference to the subject of this `Chain`."""

        return OpRef.Get(uri(self).append("subject"), key)


class SyncChain(Chain):
    """
    A :class:`Chain` which keeps track of only the current Transaction's operations,
    in order to recover from a transaction failure (e.g. if the host crashes).
    """

    __uri__ = uri(Chain) + "/sync"


Chain.Sync = SyncChain

