from ..decorators import closure, delete, get
from ..error import BadRequest
from ..generic import Map, Tuple
from ..scalar.bound import Range
from ..scalar.number import Bool, UInt
from ..scalar.ref import form_of, If, Ref
from ..state import State, Stream
from ..uri import uri, URI
from ..context import to_json

from .base import Collection
from .btree import BTree


class Schema(object):
    """A `Table` schema which comprises a primary key and value :class:`Column` s."""

    def __init__(self, key, values=[]):
        self.key = key
        self.values = values
        self.indices = []

    def __json__(self):
        return to_json([[self.key, self.values], Tuple(self.indices)])

    def columns(self):
        return self.key + self.values

    def create_index(self, name, columns):
        self.indices.append((name, columns))
        return self


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

        @closure(self)
        @get
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

        delete_row = delete(lambda cxt, key: self.delete_row(key))
        if uri(self).id():
            delete_row = closure(self)(delete_row)

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

        update_row = closure(self)(get(lambda cxt, key: self.update_row(key, values)))
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
        return self._post("", {"bounds": bounds}, Table)


def _handle_bounds(bounds):
    if bounds is None:
        return {}
    elif isinstance(bounds, State):
        return _handle_bounds(form_of(bounds))
    elif isinstance(bounds, (Ref, URI)):
        return bounds

    return {
        col: Range.from_slice(bounds[col]) if isinstance(bounds[col], slice) else bounds[col]
        for col in bounds
    }
