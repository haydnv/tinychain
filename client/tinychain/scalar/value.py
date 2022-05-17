""":class:`Value` types such as :class:`Nil`, :class:`Number`, and :class:`String`."""

from ..context import to_json
from ..interface import Compare, Order
from ..scalar.ref import deref, is_literal
from ..uri import uri, URI


from .base import Scalar
from .ref import form_of, Ref


# Scalar value types

class Value(Scalar, Compare, Order):
    """A scalar `Value` which supports equality and collation."""

    __uri__ = uri(Scalar) + "/value"

    def eq(self, other):
        if is_literal(self) and is_literal(other):
            return deref(self) == deref(other)

        return Scalar.eq(self, other)

    def gt(self, other):
        from .number import Bool
        return self._get("gt", other, Bool)

    def gte(self, other):
        from .number import Bool
        return self._get("gte", other, Bool)

    def lt(self, other):
        from .number import Bool
        return self._get("lt", other, Bool)

    def lte(self, other):
        from .number import Bool
        return self._get("lte", other, Bool)


class Nil(Value):
    """A `Value` to represent `None`."""

    __uri__ = uri(Value) + "/none"

    def __init__(self, form=None):
        if form is None:
            Value.__init__(self, {})
        else:
            Value.__init__(self, form)

    def __json__(self):
        form = form_of(self)

        if isinstance(form, (Ref, URI)):
            return to_json(form)
        else:
            return None


class Bytes(Value):
    """A binary `Value`"""

    __uri__ = uri(Value) + "/bytes"


class EmailAddress(Value):
    """An email address"""

    __uri__ = uri(Value) + "/email"


class Id(Value):
    """An identifier"""

    __uri__ = uri(Value) + "/id"

    def __json__(self):
        from .ref.helpers import form_of

        form = form_of(self)
        if isinstance(form, Ref):
            return to_json(form)
        else:
            return {str(self): []}

    def __str__(self):
        from .ref.helpers import form_of
        return str(form_of(self))


class String(Value):
    """A string."""

    __uri__ = uri(Value) + "/string"

    def render(self, params=None, **kwargs):
        if kwargs and params is not None:
            raise ValueError("String.render accepts a Map or kwargs, not both:", params, kwargs)

        if params:
            return self._post("render", params, String)
        else:
            return self._post("render", kwargs, String)


class Version(Value):
    """
    A semantic version of the form <major>.<minor>.<rev>, e.g. 1.2.34

    See https://semver.org for the full specification.
    """

    __uri__ = uri(Value) + "/version"

