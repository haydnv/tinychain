from .state import OpRef, Scalar, State
from .util import *


class Chain(State):
    __ref__ = uri(State) + "/chain"

    def set(self, value):
        return OpRef.Put(uri(self).append("subject"), None, value)

    def subject(self, key=None):
        return OpRef.Get(uri(self).append("subject"), key)


def sync_chain(initial_value):
    if not isinstance(initial_value, Scalar):
        raise ValueError(f"chain value must be a Scalar, not {initial_value}")

    dtype = type(initial_value)
    class SyncChain(Chain):
        __ref__ = uri(Chain) + "/sync"

        def subject(self, key=None) -> dtype:
            return dtype(OpRef.Get(uri(self).append("subject"), key))

    return SyncChain.init(initial_value)

