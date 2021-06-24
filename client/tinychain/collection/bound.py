from tinychain.util import to_json
from tinychain.value import Nil


class Bound(object):
    """An upper or lower bound on a :class:`Range`."""


class Ex(Bound):
    """An exclusive `Bound`."""

    def __init__(self, value):
        self.value = value

    def __json__(self):
        return to_json(["ex", self.value])


class In(Bound):
    """An inclusive `Bound`."""

    def __init__(self, value):
        self.value = value

    def __json__(self):
        return to_json(["in", self.value])


class Un(Bound):
    """An unbounded side of a :class:`Range`"""

    def __json__(self):
        return Nil


class Range(object):
    """A selection range of one or two :class:`Bound`s."""

    @staticmethod
    def from_slice(s):
        return Range(In(s.start), Ex(s.stop))

    def __init__(self, start=None, end=None):
        if start is not None and not isinstance(start, Bound):
            self.start = In(start)
        else:
            self.start = start

        if end is not None and not isinstance(end, Bound):
            self.end = In(end)
        else:
            self.end = end

    def __json__(self):
        return to_json((self.start, self.end))
