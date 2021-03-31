"""Data structures responsible for keeping track of mutations to a :class:`Value`"""

from .ref import OpRef
from .reflect import Meta
from .state import Scalar, State
from .util import *


class Chain(State, metaclass=Meta):
    """
    Data structure responsible for keeping track of mutations to a :class:`Value`.
    """

    __uri__ = uri(State) + "/chain"

    def set(self, value):
        """Update the value of this `Chain`."""

        return OpRef.Put(uri(self), None, value)


class BlockChain(Chain):
    """
    A :class:`Chain` which keeps track of the entire update history of its subject.
    """

    __uri__ = uri(Chain) + "/block"


class SyncChain(Chain):
    """
    A :class:`Chain` which keeps track of only the current transaction's operations,
    in order to recover from a transaction failure (e.g. if the host crashes).
    """

    __uri__ = uri(Chain) + "/sync"


Chain.Block = BlockChain
Chain.Sync = SyncChain

