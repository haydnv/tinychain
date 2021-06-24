from tinychain.error import BadRequest
from tinychain.ref import If
from tinychain.state import Map
from tinychain.util import uri
from tinychain.value import Bool, UInt, Nil

from .collection import Collection
from .bound import Range


class Table(Collection):
    """A `Table` defined by a primary key, values, and optional indices."""

    __uri__ = uri(Collection) + "/table"

    def __getitem__(self, key):
        """Return the row with the given key, or a :class:`NotFound` error."""

        return self._get("", key, rtype=Map)

    def contains(self, key):
        """Return `True` if this `Table` contains the given key."""

        return self._get("contains", key, rtype=Bool)

    def count(self):
        """
        Return the number of rows in this `Table`.

        To count the number of keys which match a specific range, call `table.where(range).count()`.
        """

        return self._get("count", rtype=UInt)

    def delete(self):
        """Delete all contents of this `Table`."""

        return self._delete("")

    def group_by(self, columns):
        """
        Aggregate this `Table` according to the values of the specified columns.

        If no index supports ordering by `columns`, this will raise a :class:`BadRequest` error.
        """

        return self._get("group", columns, Table)

    def insert(self, key, values=[]):
        """
        Insert the given row into this `Table`.

        If the key is already present, this will raise a :class:`BadRequest` error.
        """

        return If(
            self.contains(key),
            BadRequest("cannot insert: key already exists"),
            self._put("", key, values))

    def is_empty(self):
        """Return `True` if this table contains no rows."""

        return self._get("is_empty", rtype=Bool)

    def limit(self, limit):
        """Limit the number of rows returned from this `Table`."""

        return self._get("limit", limit, Table)

    def order_by(self, columns, reverse=False):
        """
        Set the order in which this `Table`'s rows will be iterated over.

        If no index supports the given order, this will raise a :class:`BadRequest` error.
        """

        return self._get("order", (columns, reverse), Table)

    def select(self, *columns):
        """Return a `Table` containing only the specified columns."""

        return self._get("select", columns, Table)

    def update(self, **values):
        """
        Update this `Table`\'s rows to the given values.

        To limit the range of the update, use the `where` method, e.g. `table.where(foo="bar").update(foo="baz")`.
        """

        return self._put("", None, values)

    def upsert(self, key, values=[]):
        """
        Insert the given row into this `Table`.

        If the row is already present, it will be updated with the given `values`.
        """

        return self._put("", key, values)

    def where(self, **cond):
        """
        Return a slice of this `Table` whose column values fall within the specified range.

        If there is no index which supports the given range, this will raise a :class:`BadRequest` error.
        """

        cond = {
            col: Range.from_slice(val) if isinstance(val, slice) else val
            for col, val in cond.items()}

        return self._post("", Table, **cond)
