from ..state import State
from ..util import form_of, to_json, URI

from .ref import Ref
from .value import Nil


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
            self.end = Ex(end)
        else:
            self.end = end

    def __json__(self):
        if self.start is None and self.end is None:
            return None
        else:
            return to_json((self.start, self.end))


def handle_bounds(bounds):
    if bounds is None or isinstance(bounds, Ref) or isinstance(bounds, URI):
        return bounds

    if isinstance(bounds, State):
        form = bounds
        while hasattr(form, "__form__"):
            form = form_of(form)

        if isinstance(form, tuple) or isinstance(form, list):
            bounds = form
        else:
            return bounds

    if hasattr(bounds, "__iter__"):
        bounds = tuple(bounds)
    else:
        bounds = (bounds,)

    return [Range.from_slice(x) if isinstance(x, slice) else x for x in bounds]
