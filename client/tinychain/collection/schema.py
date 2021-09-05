from tinychain.chain import Chain
from tinychain.state import Map, Tuple
from tinychain.util import to_json, uri
from tinychain.value import F32, U64


class Column(object):
    """A column in the schema of a :class:`BTree` or :class:`Table`."""

    def __init__(self, name, dtype, max_size=None):
        self.name = name
        self.dtype = dtype
        self.max_size = max_size

    def __eq__(self, other):
        return self.name == other.name and self.dtype == other.dtype and self.max_size == other.max_size

    def __json__(self):
        if self.max_size is None:
            return to_json((self.name, self.dtype))
        else:
            return to_json((self.name, self.dtype, self.max_size))

    def __repr__(self):
        if self.max_size is None:
            return f"{self.name}: column type {self.dtype}"
        else:
            return f"{self.name}: column type {self.dtype}, max size {self.max_size}"


class Edge(object):
    """
    A directed edge between two node columns in a :class:`Graph`.

    The format of `from_node` and `to_node` is "<table name>.<column name>", e.g. "users.user_id".

    If the `cascade` attribute is set to `True`, deleting a source row will automatically delete all foreign
    rows which depend on it. If `cascade` is `False`, deleting a source row when foreign keys are still present
    will raise a :class:`BadRequest` error.
    """

    def __init__(self, from_node, to_node, cascade=False):
        assert '.' in from_node
        assert '.' in to_node

        self.cascade = cascade
        self.from_table, from_column = from_node.split('.')
        self.to_table, to_column = to_node.split('.')

        if from_column == to_column:
            self.column = from_column
        else:
            raise ValueError(f"edge columns must have the same name: {from_column}, {to_column}")


class BTree(object):
    """A :class:`BTree` schema which comprises a tuple of :class:`Column` s."""

    def __init__(self, *columns):
        self.columns = columns

    def __json__(self):
        return to_json(self.columns)


class Graph(object):
    """A :class:`Graph` schema which comprises a set of :class:`Table` s and edges between :class:`Table` columns."""

    def __init__(self, chain):
        if not issubclass(chain, Chain):
            raise ValueError(f"default Chain type must be a subclass of Chain, not {chain}")

        self.chain = chain
        self.tables = {}
        self.edges = {}

    def create_table(self, name, schema):
        """Add a :class:`Table` to this `Graph`."""

        self.tables[name] = schema
        return self

    def create_edge(self, name, edge):
        """Add an :class:`Edge` between tables in this `Graph`."""

        assert edge.from_table in self.tables
        from_table = self.tables[edge.from_table]

        assert edge.to_table in self.tables
        to_table = self.tables[edge.to_table]

        if from_table.key != [Column(edge.column, U64)]:
            raise ValueError(f"invalid foreign key column: {edge.from_table}.{edge.column} (key is {from_table.key})")

        [pk] = [col for col in from_table.key if col.name == edge.column]

        if edge.from_table != edge.to_table:
            if len(to_table.key) != 1:
                raise ValueError("the primary key of a Graph node type must be a single U64 column, not", to_table.key)

            [fk] = [col for col in to_table.values if col.name == edge.column]
            if pk != fk:
                raise ValueError(
                    f"primary key {edge.from_table}.{pk.name} does not match foreign key {edge.to_table}.{fk.name}")

            has_index = False
            for (_name, columns) in to_table.indices:
                if columns and columns[0] == edge.column:
                    has_index = True

            if not has_index:
                raise ValueError(f"there is no index on {edge.to_table} to support the foreign key on {edge.column}")
        elif to_table.key != [pk]:
            raise ValueError(f"Graph node {edge.to_table} self-reference must be to the primary key, not {edge.column}")

        self.edges[name] = edge
        return self


class Table(object):
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
