""":class:`Value` types such as :class:`Nil`, :class:`Number`, and :class:`String`."""

from ..util import form_of, to_json, uri, URI

from .base import Scalar
from .ref import If, Ref


# Scalar value types

class Value(Scalar):
    """A scalar `Value` which supports equality and collation."""

    __uri__ = uri(Scalar) + "/value"

    @classmethod
    def max(cls, l, r):
        """Return `l`, or `r`, whichever is greater."""

        return cls(If(l >= r, l, r))

    @classmethod
    def min(cls, l, r):
        """Return `l`, or `r`, whichever is lesser."""

        return cls(If(l <= r, l, r))

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __eq__(self, other):
        return self.eq(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.gte(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.lte(other)

    def eq(self, other):
        """Returns `true` if `self` is equal to `other`."""

        from .number import Bool
        return self._get("eq", other, Bool)

    def ne(self, other):
        """Returns `true` if `self` is not equal to `other`."""

        from .number import Bool
        return self._get("ne", other, Bool)

    def gt(self, other):
        """Return true if `self` is greater than `other`."""

        from .number import Bool
        return self._get("gt", other, Bool)

    def gte(self, other):
        """Return true if `self` is greater than or equal to `other`."""

        from .number import Bool
        return self._get("gte", other, Bool)

    def lt(self, other):
        """Return true if `self` is less than `other`."""

        from .number import Bool
        return self._get("lt", other, Bool)

    def lte(self, other):
        """Return true if `self` is less than or equal to `other`."""

        from .number import Bool
        return self._get("lte", other, Bool)


class Nil(Value):
    """A `Value` to represent `None`."""

    __uri__ = uri(Value) + "/none"

    def __json__(self):
        form = form_of(self)

        if isinstance(form, Ref) or isinstance(form, URI):
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
        form = form_of(self)
        if isinstance(form, Ref):
            return to_json(form)
        else:
            return {str(self): []}

    def __str__(self):
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

