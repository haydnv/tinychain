from tinychain.state import State
from tinychain.util import uri


# A stream of `State` s

class Stream(State):
    """A stream of states which supports functional methods like `fold` and `map`."""

    __uri__ = uri(State) + "/stream"

    @classmethod
    def range(cls, range):
        """Return a stream of numbers in the given `range`.

        `range` can be a positive number, a 2-Tuple like `(start, stop)`, or a 3-Tuple like `(start, stop, step)`
        """

        from .ref import Get
        return cls(Get(uri(cls) + "/range", range))

    def aggregate(self):
        return self._get("aggregate", rtype=Stream)

    def filter(self, op):
        """Return a new `Stream` containing only those elements of this `Stream` where the given `op` returns `True`."""

        return self._post("filter", {"op": op}, Stream)

    def first(self):
        """Return the first item in this `Stream`, or `Nil` if the `Stream` is empty."""

        return self._get("first", rtype=State)

    def flatten(self):
        """Flatten a `Stream` of `Stream` s into a single `Stream` of their component elements."""

        return self._get("flatten", rtype=Stream)

    def for_each(self, op):
        """Run the given `op` for each item in this `Stream`, then return the last result.

        This is useful when you need to execute an `op` for its side-effects and not its return value.
        """

        rtype = op.rtype if hasattr(op, "rtype") else State
        return self._post("for_each", {"op": op}, rtype)

    def fold(self, item_name, initial_state, op):
        """Run the given `op` for each item in this `Stream` along with the previous result.

        `op` must be a POST Op. The stream item to handle will be passed with the given `item_name` as its name.
        """

        rtype = type(initial_state) if isinstance(initial_state, State) else State
        return self._post("fold", {"item_name": item_name, "value": initial_state, "op": op}, rtype)

    def map(self, op):
        """Return a new `Stream` whose items are the results of running `op` on each item of this `Stream`."""

        return self._post("map", {"op": op}, Stream)
