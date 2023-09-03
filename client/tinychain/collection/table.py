from __future__ import annotations

from typing import TYPE_CHECKING, Type

from ..error import BadRequest
from ..generic import Map, Tuple
from ..json import to_json
from ..scalar.bound import Range
from ..scalar.number import Bool, UInt
from ..scalar.ref import If, Ref, form_of
from ..state import State
from ..uri import URI
from .base import Collection, Column

if TYPE_CHECKING:
    from ..service import Model


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

    __uri__ = URI(Collection) + "/table"

    def __getitem__(self, key):
        """Return the row with the given key, or a :class:`NotFound` error."""

        return self._get("", key, rtype=Map)

    # TODO: re-enable this functionality after implementing Graph slicing
    def aggregate(self, columns, fn):
        """
        Apply the given callback to slices of this `Table` grouped by the given columns.

        Returns a stream of tuples of the form (<unique column values>, <callback result>).

        Example: `orders.aggregate(["customer_id", "product_id"], Table.count)`
        """

        raise NotImplementedError("Table.aggregate has been temporarily disabled")

    def contains(self, key):
        """Return `True` if this `Table` contains the given key."""

        return self._get("contains", key, rtype=Bool)

    def columns(self):
        """Return the column schema of this `Table` as a :class:`Tuple`."""

        return self._get("columns", rtype=Tuple)

    def count(self):
        """Return the number of rows in the given slice of this `Table`."""

        return self._get("count", rtype=UInt)

    def delete(self, key):
        """Delete the row of this `Table` with the given `key`."""

        return self._delete("", key)

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

    def select(self, columns):
        """Return a `Table` containing only the specified columns."""

        return self._get("select", columns, Table)

    def truncate(self):
        """Delete all rows in this :class:`Table`."""

        return self.delete("")

    def update(self, **values):
        """Update the rows of this table with the given `values`."""

        return self._put("", values)

    def upsert(self, key, values):
        """
        Insert the given row into this `Table`.

        If the row is already present, it will be updated with the given `values`.
        """

        return self._put("", key, values)

    def where(self, **bounds):
        """
        Return a slice of this `Table` whose column values fall within the specified range.

        If there is no index which supports the given range, this will raise a :class:`BadRequest` error.
        """

        if not bounds:
            return self

        parent = self
        bounds = handle_bounds(bounds)

        class WriteableView(Table):
            def delete(self, key):
                return RuntimeError(f"cannot delete the row at {key} from a slice {self} of a table {parent}")

            def update(self, **values):
                return parent._put("", [(col, bounds[col]) for col in bounds], values)

            def upsert(self, key, values):
                return RuntimeError(f"cannot upsert ({key}, {values}) into a slice {self} of a table {parent}")

            def truncate(self):
                return parent._delete("", [(col, bounds[col]) for col in bounds])

        return self._get("", [(col, bounds[col]) for col in bounds], WriteableView)


def handle_bounds(bounds):
    if bounds is None:
        return {}
    elif isinstance(bounds, State):
        return handle_bounds(form_of(bounds))
    elif isinstance(bounds, (Ref, URI)):
        return bounds

    return {
        col: Range.from_slice(bounds[col]) if isinstance(bounds[col], slice) else bounds[col]
        for col in bounds
    }


# TODO: move to the graph package
def create_schema(modelclass: Type[Model]) -> Schema:
    """
    Create a table schema for the given model.

    A key for the table is auto generated using the `class_name` function, then suffixed with '_id'.
    Each attribute of the model will be considered as a column if it is of type :class:`Column` or :class:`Model`.
    """

    values = []
    indices = []
    base_attributes = set()

    for b in modelclass.__bases__:
        base_attributes |= set(dir(b))

    for f in base_attributes ^ set(dir(modelclass)):
        attr = getattr(modelclass, f)
        if isinstance(attr, Column):
            values.append(attr)
        else:
            try:
                from ..service import Model, class_name
                assert issubclass(attr, Model)
                values.append(*attr.key())
                indices.append((class_name(attr), [attr.key()[0].name]))
            except (TypeError, AssertionError):
                continue

    schema = Schema(modelclass.key(), values)
    for i in indices:
        schema.create_index(*i)

    return schema
