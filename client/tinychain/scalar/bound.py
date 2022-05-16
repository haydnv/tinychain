from ..state import State
from ..uri import URI
from ..context import to_json

from .ref import form_of, Ref


class Bound(object):
    """An upper or lower bound on a :class:`Range`."""


class Ex(Bound):
    """An exclusive `Bound`."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ex {self.value}"

    def __json__(self):
        return to_json(["ex", self.value])


class In(Bound):
    """An inclusive `Bound`."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"in {self.value}"

    def __json__(self):
        return to_json(["in", self.value])


class Range(object):
    """A selection range of one or two :class:`Bound`s."""

    def __repr__(self):
        start, end = None, None

        if isinstance(self.start, In):
            start = f"[{repr(self.start.value)}"
        if isinstance(self.start, Ex):
            start = f"({repr(self.start.value)}"

        if isinstance(self.end, In):
            end = f"{repr(self.end.value)}]"
        if isinstance(self.end, Ex):
            end = f"{repr(self.end.value)})"

        if start and end:
            return f"{start}:{end}"
        elif start:
            return f"{start}:"
        elif end:
            return f":{end}"
        else:
            return ":"

    @staticmethod
    def from_slice(s):
        return Range(In(s.start), Ex(s.stop))

    def to_slice(self):
        start = self.start.value if self.start else None
        end = self.end.value if self.end else None
        return slice(form_of(start), form_of(end))

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
    if bounds is None or isinstance(bounds, (Ref, URI)):
        return bounds

    if isinstance(bounds, State):
        form = bounds
        while hasattr(form, "__form__"):
            form = form_of(form)

        if isinstance(form, (tuple, list)):
            bounds = form
        else:
            return bounds

    if hasattr(bounds, "__iter__"):
        bounds = tuple(bounds)
    else:
        bounds = (bounds,)

    return [Range.from_slice(x) if isinstance(x, slice) else x for x in bounds]
