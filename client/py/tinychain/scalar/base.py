from ..interface import Equality
from ..json import to_json
from ..state import State
from ..uri import URI


class Scalar(State, Equality):
    """
    An immutable :class:`State` which always resides entirely in the host's memory.

    Do not subclass :class:`Scalar` directly. Use :class:`Value` instead.
    """

    __uri__ = URI(State) + "/scalar"

    def __json__(self):
        from .ref.functions import form_of
        return to_json(form_of(self))

    def eq(self, other):
        from .number import Bool
        return self._get("eq", other, Bool)

    def ne(self, other):
        from .number import Bool
        return self._get("ne", other, Bool)
