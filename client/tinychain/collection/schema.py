from tinychain.state import Map, Tuple
from tinychain.util import to_json, uri
from tinychain.value import F32


class Column(object):
    """A column in the schema of a :class:`BTree` or :class:`Table`."""

    def __init__(self, name, dtype, max_size=None):
        self.name = name
        self.dtype = dtype
        self.max_size = max_size

    def __json__(self):
        if self.max_size is None:
            return to_json((self.name, self.dtype))
        else:
            return to_json((self.name, self.dtype, self.max_size))


class BTree(object):
    """A :class:`BTree` schema which comprises a tuple of :class:`Column` s."""

    def __init__(self, *columns):
        self.columns = columns

    def __json__(self):
        return to_json(self.columns)


class Graph(object):
    """A :class:`Graph` schema which comprises a set of :class:`Table` s and edges between :class:`Table` columns."""

    def __init__(self):
        self.tables = {}
        self.edges = {}

    def add_table(self, name, schema):
        """Add a :class:`Table` to this `Graph`."""

        self.tables[name] = schema
        return self

    def add_edge(self, name, from_node, to_node):
        """Add an edge between tables in this `Graph`."""

        self.edged[name] = (from_node, to_node)
        return self


class Table(object):
    """A `Table` schema which comprises a primary key and value :class:`Column` s."""

    def __init__(self, key, values=[]):
        self.key = key
        self.values = values
        self.indices = []

    def __json__(self):
        return to_json([[self.key, self.values], Tuple(self.indices)])

    def add_index(self, name, columns):
        self.indices.append((name, columns))
        return self


class Tensor(object):
    """
    A `Tensor` schema which comprises a shape and data type.

    The data type must be a subclass of `Number` and defaults to `F32`.
    """

    def __init__(self, shape, dtype=F32):
        self.shape = shape
        self.dtype = dtype

    def __json__(self):
        return to_json([self.shape, self.dtype])
