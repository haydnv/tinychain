from tinychain.collection.btree import BTree
from tinychain.decorators import closure, delete_op, get_op, post_op
from tinychain.error import BadRequest
from tinychain.ref import Delete, If, Ref
from tinychain.state import Map, Tuple, State, Stream
from tinychain.util import form_of, uri, Context, URI
from tinychain.value import Bool, UInt, Nil

from .collection import Collection
from .bound import Range


class Table(Collection):
    """A `Table` defined by a primary key, values, and optional indices."""

    __uri__ = uri(Collection) + "/table"

    def __getitem__(self, key):
        """Return the row with the given key, or a :class:`NotFound` error."""

        return self._get("", key, rtype=Map)

    def aggregate(self, columns, fn):
        """
        Apply the given callback to slices of this `Table` grouped by the given columns.

        Returns a stream of tuples of the form (<unique column values>, <callback result>).

        Example: `orders.aggregate(["customer_id", "product_id"], Table.count)`
        """

        @closure
        @get_op
        def group(cxt, key: Tuple) -> Tuple:
            cxt.where = Tuple(columns).zip(key).cast(Map)
            cxt.slice = self.where(cxt.where)
            return key, fn(cxt.slice)

        return self.group_by(columns).map(group)

    def contains(self, key):
        """Return `True` if this `Table` contains the given key."""

        return self._get("contains", key, rtype=Bool)

    def columns(self):
        """Return the column schema of this `Table` as a :class:`Tuple`."""

        return self._get("columns", rtype=Tuple)

    def count(self, where=None):
        """Return the number of rows in the given slice of this `Table` (or the entire `Table` if no bounds are given)."""

        if where is None:
            return self._get("count", rtype=UInt)
        else:
            return self.where(where).count()

    def delete(self, where={}):
        """
        Delete all contents of this `Table` matching the specified where clause.

        If no where clause is specified, all contents of this `Table` will be deleted.
        """

        delete_row = closure(delete_op(lambda cxt, key: self.delete_row(key)))
        to_delete = self.where(where) if where else self
        return to_delete.select(self.key_names()).rows().for_each(delete_row)

    def delete_row(self, key):
        """Delete the row with the given key from this `Table`, if it exists."""

        return self._delete("", key)

    def group_by(self, columns):
        """Return a :class:`Stream` of the unique values of the given columns."""

        return self.order_by(columns).select(columns).rows().aggregate()

    def index(self):
        """Build a :class:`BTree` index with the values of the given columns."""

        return BTree.copy_from(self.columns(), self.rows())

    def insert(self, key, values=[]):
        """
        Insert the given row into this `Table`.

        If the key is already present, this will raise a :class:`BadRequest` error.
        """

        return If(
            self.contains(key),
            BadRequest("cannot insert: key already exists"),
            self.upsert(key, values))

    def is_empty(self):
        """Return `True` if this table contains no rows."""

        return self._get("is_empty", rtype=Bool)

    def key_columns(self):
        """Return the schema of the key columns of this `Table`."""

        return self._get("key_columns", rtype=Tuple)

    def key_names(self):
        """Return the `Id` s of the key columns of this `Table`."""

        return self._get("key_names", rtype=Tuple)

    def limit(self, limit):
        """Limit the number of rows returned from this `Table`."""

        return self._get("limit", limit, Table)

    def order_by(self, columns, reverse=False):
        """
        Set the order in which this `Table`'s rows will be iterated over.

        If no index supports the given order, this will raise a :class:`BadRequest` error.
        """

        return self._get("order", (columns, reverse), Table)

    def rows(self, where={}):
        """Return a :class:`Stream` of the rows in this `Table`."""

        where = _handle_bounds(where)
        return self._post("rows", where, Stream)

    def select(self, columns):
        """Return a `Table` containing only the specified columns."""

        return self._get("select", columns, Table)

    def update(self, values, where={}):
        """Update the specified rows of this table with the given `values`."""

        update_row = closure(get_op(lambda cxt, key: self.update_row(key, values)))
        return self.where(where).select(self.key_names()).index().keys().for_each(update_row)

    def update_row(self, key, values):
        """Update the specified row with the given `values`."""

        return self._put("", key, values)

    def upsert(self, key, values):
        """
        Insert the given row into this `Table`.

        If the row is already present, it will be updated with the given `values`.
        """

        return self._put("", key, values)

    def where(self, bounds):
        """
        Return a slice of this `Table` whose column values fall within the specified range.

        If there is no index which supports the given range, this will raise a :class:`BadRequest` error.
        """

        bounds = _handle_bounds(bounds)
        return self._post("", Map(bounds=bounds), Table)


def _handle_bounds(bounds):
    if bounds is None:
        return {}
    elif isinstance(bounds, State):
        return _handle_bounds(form_of(bounds))
    elif isinstance(bounds, Ref) or isinstance(bounds, URI):
        return bounds

    return {
        col: Range.from_slice(val) if isinstance(val, slice) else val
        for col, val in dict(bounds).items()
    }
