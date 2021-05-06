"""Data structures responsible for storing a collection of :class:`Value`\s"""

from .error import BadRequest, NotFound
from .ref import If, OpRef
from .reflect import Meta
from .state import Map, State
from .util import *
from .value import Bool, Nil, UInt, Value


class Bound(object):
    """An upper or lower bound on a :class:`Range`."""

    pass


class Ex(Bound):
    """An exclusive `Bound`."""

    def __init__(self, value):
        self.value = value

    def __json__(self):
        return to_json({"ex": self.value})


class In(Bound):
    """An inclusive `Bound`."""

    def __init__(self, value):
        self.value = value

    def __json__(self):
        return to_json({"in": self.value})


class Un(Bound):
    """An unbounded side of a :class:`Range`"""

    def __json__(self):
        return Nil


Bound.Ex = Ex
Bound.In = In


class Range(object):
    """A selection range of one or two :class:`Bound`\s."""

    @staticmethod
    def from_slice(s):
        return Range(Bound.In(s.start), Bound.Ex(s.stop))

    def __init__(self, start=None, end=None):
        if start is not None and not isinstance(start, Bound):
            self.start = Bound.In(start)
        else:
            self.start = start

        if end is not None and not isinstance(end, Bound):
            self.end = Bound.In(end)
        else:
            self.end = end

    def __json__(self):
        return to_json((self.start, self.end))


class Column(object):
    """A column in the schema of a :class:`BTree`."""

    def __init__(self, name, dtype, max_size=None):
        self.name = str(name)
        self.dtype = uri(dtype)
        self.max_size = max_size

    def __json__(self):
        if self.max_size is None:
            return to_json((self.name, str(self.dtype)))
        else:
            return to_json((self.name, str(self.dtype), self.max_size))


class Collection(State, metaclass=Meta):
    """Data structure responsible for storing a collection of :class:`Value`\s."""

    __uri__ = uri(State) + "/collection"


class BTree(Collection):
    """A `BTree` with a schema of named, :class:`Value`-typed :class:`Column`\s."""

    __uri__ = uri(Collection) + "/btree"

    class Schema(object):
        """A `BTree` schema which comprises a tuple of :class:`Column`\s."""

        def __init__(self, *columns):
            self.columns = columns

        def __json__(self):
            return to_json(self.columns)

    def __getitem__(self, prefix):
        """
        Return a slice of this `BTree` containing all keys which begin with the given prefix.
        """

        if not isinstance(prefix, tuple):
            prefix = (prefix,)

        prefix = [Range.from_slice(k) if isinstance(k, slice) else k for k in prefix]

        if any(isinstance(k, Range) for k in prefix):
            return self._post("", BTree, **{"range": prefix})
        else:
            return self._get("", prefix, BTree)

    def count(self):
        """
        Return the number of keys in this `BTree`.

        To count the number of keys beginning with a specific prefix,
        call `btree[prefix].count()`.
        """

        return self._get("count", rtype=UInt)

    def delete(self):
        """
        Delete the contents of this `BTree`.

        To delete all keys beginning with a specific prefix, call
        `btree[prefix].delete()`.
        """

        return self._delete("")

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
        Return a slice of this `BTree` with the same range with its keys in reverse order.
        """

        return self._get("", (None, True), BTree)


class Table(Collection):
    """A `Table` defined by a primary key, values, and optional indices."""

    __uri__ = uri(Collection) + "/table"

    class Schema(object):
        """A `Table` schema which comprises a primary key and value :class:`Column`\s."""

        def __init__(self, key, values=[], indices={}):
            self.key = key
            self.values = values
            self.indices = indices

        def __json__(self):
            return to_json([[self.key, self.values], list(self.indices.items())])

    def __getitem__(self, key):
        """Return the row with the given key, or a :class:`NotFound` error."""

        return Map(self._get("", key))

    def contains(self, key):
        """Return `True` if this `Table` contains the given key."""

        return Bool(self._get("contains", key))

    def count(self):
        """
        Return the number of rows in this `Table`.

        To count the number of keys which match a specific range,
        call `table.where(range).count()`.
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

        If no index supports the given order, this will raise a
        :class:`BadRequest` error.
        """

        return self._get("order", (columns, reverse), Table)

    def select(self, *columns):
        """Return a `Table` containing only the specified columns."""

        return self._get("select", columns, Table)

    def upsert(self, key, values=[]):
        """
        Insert the given row into this `Table`.

        If the row is already present, it will be updated with the given `values`.
        """

        return self._put("", key, values)

    def where(self, **cond):
        """
        Return a slice of this `Table` whose column values fall within the specified range.

        If there is no index which supports the given range, this will raise 
        a :class:`BadRequest` error.
        """

        cond = {
            col: Range.from_slice(val) if isinstance(val, slice) else val
            for col, val in cond.items()}

        return self._post("", Table, **cond)

