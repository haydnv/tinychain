from .state import State


class Chain(State):
    PATH = State.PATH + "/chain"


class SyncChain(Chain):
    Path = Chain.PATH = "/sync"


Chain.Sync = SyncChain

