"""A `BTree` with a schema of named, :class:`Value`-typed :class:`Column` s."""

from ..generic import Map, Tuple
from ..json import to_json
from ..scalar.bound import Range
from ..scalar.number import UInt
from ..scalar.ref import form_of, Ref
from ..state import State
from ..uri import URI

from .base import Collection


class Schema(object):
    """A :class:`BTree` schema which comprises a tuple of :class:`Column` s."""

    def __init__(self, columns):
        self.columns = columns

    def __json__(self):
        return to_json(self.columns)


class BTree(Collection):
    """A `BTree` with a schema of named, :class:`Value`-typed :class:`Column` s."""

    __uri__ = URI(Collection) + "/btree"

    def __getitem__(self, prefix):
        """
        Return a slice of this `BTree` containing all keys which begin with the given prefix.
        """

        if isinstance(prefix, list):
            prefix = tuple(prefix)
        elif prefix is not None and not isinstance(prefix, (Tuple, tuple)):
            prefix = (prefix,)

        range = _handle_range(prefix)
        return self._post("", {"range": range}, BTree)

    def count(self, range=None):
        """
        Return the number of keys in this `BTree`.

        To count the number of keys beginning with a specific prefix, call `btree[prefix].count()`.
        """

        range = _handle_range(range)
        return self._get("count", range, UInt)

    def delete(self, prefix=None):
        """
        Delete the contents of this `BTree` within the specified range.

        If no range is specified, the entire contents of this `BTree` will be deleted.
        """

        if isinstance(prefix, list):
            prefix = tuple(prefix)
        elif prefix is not None and not isinstance(prefix, (Tuple, tuple)):
            prefix = (prefix,)

        prefix = _handle_range(prefix)
        return self._delete("", prefix)

    def first(self):
        """
        Return the first row in this `BTree`.

        If there are no rows, this will raise a :class:`NotFoundError`.
        """

        return self._get("first", rtype=Map)

    def insert(self, key):
        """
        Insert the given key into this `BTree`.

        If the key is already present, this is a no-op.
        """

        return self._put("", value=key)

    def reverse(self):
        """
        Return a slice of this `BTree` with the same range but with its keys in reverse order.
        """

        return self._get("reverse", rtype=BTree)


def _handle_range(range):
    if range is None or isinstance(range, (Ref, URI)):
        return range

    if isinstance(range, State):
        form = form_of(range)
        if isinstance(form, (list, tuple)):
            range = form
        else:
            return range

    return [Range.from_slice(k) if isinstance(k, slice) else k for k in range]
