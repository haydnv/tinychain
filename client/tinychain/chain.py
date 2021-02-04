from .state import State
from .util import *


class Chain(State):
    __ref__ = uri(State) + "/chain"


class SyncChain(Chain):
    __ref__ = uri(Chain) + "/sync"


Chain.Sync = SyncChain

