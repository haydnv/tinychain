"""Data structures responsible for storing a collection of :class:`Value`\s"""

from .error import BadRequest, NotFound
from .ref import If, OpRef
from .reflect import Meta
from .state import Map, State
from .util import *
from .value import Bool, F32, Nil, Number, UInt, Value


class Bound(object):
    """An upper or lower bound on a :class:`Range`."""


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

        return self._post("", BTree, **{"range": prefix})

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
        Return a slice of this `BTree` with the same range but with its keys in reverse order.
        """

        return self._get("reverse", rtype=BTree)


# TODO: add `update` method
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

    def update(self, **values):
        """
        Update this `Table`\'s rows to the given values.

        To limit the range of this update, use the `where` method, e.g.
        `table.where(foo="bar").update(foo="baz")`.
        """

        return self._post("update", Nil, **values)

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


class Tensor(Collection):
    "An n-dimensional array of numbers."

    __uri__ = uri(Collection) + "/tensor"

    class Schema(object):
        """
        A `Tensor` schema which comprises a shape and data type.

        The data type must be a subclass of `Number` and defaults to `F32`.
        """

        def __init__(self, shape, dtype=F32):
            self.shape = shape
            self.dtype = dtype

        def __json__(self):
            return to_json([self.shape, str(uri(self.dtype))])

    def __getitem__(self, bounds):
        if not isinstance(bounds, tuple):
            bounds = tuple(bounds)

        bounds = [
            Range.from_slice(x) if isinstance(x, slice)
            else x for x in bounds]

        return self._post("", Tensor, bounds=bounds)

    def __add__(self, other):
        return self.add(other)

    def __div__(self, other):
        return self.div(other)

    def __truediv__(self, other):
        return self.div(other)

    def __mul__(self, other):
        return self.mul(other)

    def __sub__(self, other):
        return self.sub(other)

    def abs(self):
        return self._get("abs", rtype=Tensor)

    def add(self, other):
        return self._post("add", Tensor, r=other)

    def all(self):
        """Return `True` if all elements in this `Tensor` are nonzero."""

        return self._get("all", rtype=Tensor)

    def any(self):
        """Return `True` if any element in this `Tensor` are nonzero."""

        return self._get("any", rtype=Tensor)

    def div(self, other):
        """Divide this `Tensor` by another, broadcasting if necessary."""

        return self._post("div", Tensor, r=other)

    def eq(self, other):
        """Return a boolean `Tensor` with element-wise equality values."""

        return self._post("eq", Tensor, r=other)

    def gt(self, other):
        """Return a boolean `Tensor` with element-wise greater-than values."""

        return self._post("gt", Tensor, r=other)

    def gte(self, other):
        """Return a boolean `Tensor` with element-wise greater-or-equal values."""

        return self._post("gte", Tensor, r=other)

    def lt(self, other):
        """Return a boolean `Tensor` with element-wise less-than values."""

        return self._post("lt", Tensor, r=other)

    def lte(self, other):
        """Return a boolean `Tensor` with element-wise less-or-equal values."""

        return self._post("lte", Tensor, r=other)

    def logical_and(self, other):
        """Return a boolean `Tensor` with element-wise logical and values."""

        return self._post("and", Tensor, r=other)

    def logical_not(self):
        """Return a boolean `Tensor` with element-wise logical not values."""

        return self._get("not", rtype=Tensor)

    def logical_or(self, other):
        """Return a boolean `Tensor` with element-wise logical or values."""

        return self._post("or", Tensor, r=other)

    def logical_xor(self, other):
        """Return a boolean `Tensor` with element-wise logical xor values."""

        return self._post("xor", Tensor, r=other)

    def mul(self, other):
        """Multiply this `Tensor` by another, broadcasting if necessary."""

        return self._post("mul", Tensor, r=other)

    def ne(self, other):
        """Return a boolean `Tensor` with element-wise not-equal values."""

        return self._post("ne", Tensor, r=other)

    def sub(self, other):
        """Subtract another `Tensor` from this one, broadcasting if necessary."""

        return self._post("sub", Tensor, r=other)

    def write(self, bounds, value):
        """
        Write a `Tensor` or `Number` to the given slice of this one.

        If `bounds` is `None`, this entire `Tensor` will be overwritten.
        """

        return self._put("", bounds, value)


class DenseTensor(Tensor):
    "An n-dimensional array of numbers stored as sequential blocks."

    __uri__ = uri(Tensor) + "/dense"

    @classmethod
    def arange(cls, shape, start, stop):
        """
        Return a `DenseTensor` with the given shape containing a range of numbers
        evenly distributed between `start` and `stop`.
        """

        return cls(OpRef.Get(uri(cls) + "/range", (shape, start, stop)))

    @classmethod
    def constant(cls, shape, value):
        """Return a `DenseTensor` filled with the given `value`."""

        return cls(OpRef.Get(uri(cls) + "/constant", (shape, value)))

    @classmethod
    def ones(cls, shape, dtype=F32):
        """
        Return a `DenseTensor` filled with ones.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.constant(shape, dtype(1))

    @classmethod
    def zeros(cls, shape, dtype=F32):
        """
        Return a `DenseTensor` filled with ones.

        If `dtype` is not specified, the data type will be :class:`F32`.
        """

        return cls.constant(shape, dtype(0))


Tensor.Dense = DenseTensor

