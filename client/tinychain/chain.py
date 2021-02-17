from .state import OpRef, Scalar, State
from .util import *


class Chain(State):
    __ref__ = uri(State) + "/chain"

    def __json__(self):
        return {str(uri(type(self))): [to_json(ref(self))]}

    def set(self, value):
        return OpRef.Put(uri(self).append("subject"), None, value)

    def subject(self, key=None):
        return OpRef.Get(uri(self).append("subject"), key)


class SyncChain(Chain):
    __ref__ = uri(Chain) + "/sync"


Chain.Sync = SyncChain

