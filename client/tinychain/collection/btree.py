"""A `BTree` with a schema of named, :class:`Value`-typed :class:`Column` s."""

from tinychain.state import Map
from tinychain.util import uri
from tinychain.value import UInt

from .collection import Collection
from .bound import Range


class BTree(Collection):
    """A `BTree` with a schema of named, :class:`Value`-typed :class:`Column` s."""

    __uri__ = uri(Collection) + "/btree"

    def __getitem__(self, prefix):
        """
        Return a slice of this `BTree` containing all keys which begin with the given prefix.
        """

        if not isinstance(prefix, tuple):
            prefix = (prefix,)

        prefix = [Range.from_slice(k) if isinstance(k, slice) else k for k in prefix]

        return self._post("", BTree, **{"range": prefix})

    def count(self):
        """
        Return the number of keys in this `BTree`.

        To count the number of keys beginning with a specific prefix,
        call `btree[prefix].count()`.
        """

        return self._get("count", rtype=UInt)

    def delete(self, key=None):
        """
        Delete the contents of this `BTree` beginning with the specified prefix.

        If no prefix is specified, the entire contents of this `BTree` will be deleted.
        """

        return self._delete("", key)

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
