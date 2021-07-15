from tinychain.ref import Put
from tinychain.reflect.meta import Meta
from tinychain.state import State
from tinychain.util import uri


class Collection(State, metaclass=Meta):
    """Data structure responsible for storing a collection of :class:`Value`s."""

    __uri__ = uri(State) + "/collection"

    @classmethod
    def load(cls, schema, data):
        """
        Load a :class:`Collection` from the given `data`.

        Example:
            .. highlight:: python
            .. code-block:: python

                elements = range(10)
                dense = tc.tensor.Dense([2, 5], tc.I32, elements)
        """

        return cls(Put(cls, schema, data))
