from tinychain.reflect.meta import Meta
from tinychain.state import State
from tinychain.util import uri


class Collection(State, metaclass=Meta):
    """Data structure responsible for storing a collection of :class:`Value`s."""

    __uri__ = uri(State) + "/collection"
