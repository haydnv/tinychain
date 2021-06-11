from tinychain.reflect.meta import Meta
from tinychain.state import State
from tinychain.util import to_json, uri
from tinychain.value import Nil


class Bound(object):
    """An upper or lower bound on a :class:`Range`."""


class Ex(Bound):
    """An exclusive `Bound`."""

    def __init__(self, value):
        self.value = value

    def __json__(self):
        return to_json({"ex": self.value})


class In(Bound):
    """An inclusive `Bound`."""

    def __init__(self, value):
        self.value = value

    def __json__(self):
        return to_json({"in": self.value})


class Un(Bound):
    """An unbounded side of a :class:`Range`"""

    def __json__(self):
        return Nil


Bound.Ex = Ex
Bound.In = In
Bound.Un = Un


class Range(object):
    """A selection range of one or two :class:`Bound`\s."""

    @staticmethod
    def from_slice(s):
        return Range(Bound.In(s.start), Bound.Ex(s.stop))

    def __init__(self, start=None, end=None):
        if start is not None and not isinstance(start, Bound):
            self.start = Bound.In(start)
        else:
            self.start = start

        if end is not None and not isinstance(end, Bound):
            self.end = Bound.In(end)
        else:
            self.end = end

    def __json__(self):
        return to_json((self.start, self.end))


class Column(object):
    """A column in the schema of a :class:`BTree`."""

    def __init__(self, name, dtype, max_size=None):
        self.name = str(name)
        self.dtype = uri(dtype)
        self.max_size = max_size

    def __json__(self):
        if self.max_size is None:
            return to_json((self.name, str(self.dtype)))
        else:
            return to_json((self.name, str(self.dtype), self.max_size))


class Collection(State, metaclass=Meta):
    """Data structure responsible for storing a collection of :class:`Value`\s."""

    __uri__ = uri(State) + "/collection"
