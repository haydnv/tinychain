from .ref import OpRef
from .state import Scalar, State
from .util import *


class Chain(State):
    __uri__ = uri(State) + "/chain"

    def set(self, value):
        return OpRef.Put(uri(self).append("subject"), None, value)

    def subject(self, key=None):
        return OpRef.Get(uri(self).append("subject"), key)


class SyncChain(Chain):
    __uri__ = uri(Chain) + "/sync"


Chain.Sync = SyncChain

