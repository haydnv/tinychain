from .state import State
from .util import *


class Chain(State):
    PATH = State.PATH + "/chain"

    def __getattr__(self, name):
        attr = getattr(spec(self), name)
        return ref(attr, name)

class SyncChain(Chain):
    PATH = Chain.PATH + "/sync"


Chain.Sync = SyncChain

