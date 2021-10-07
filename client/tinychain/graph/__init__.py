from tinychain.chain import Chain, Sync
from tinychain.cluster import Cluster
from tinychain.collection import Column
from tinychain.collection.table import Table
from tinychain.collection.tensor import einsum, Sparse
from tinychain.error import BadRequest
from tinychain.decorators import closure, get_op, post_op, put_op, delete_op
from tinychain.ref import After, Get, If, MethodSubject, While, With
from tinychain.state import Map, Tuple
from tinychain.util import uri, Context, URI
from tinychain.value import Bool, Nil, I64, U64, String

from .edge import Edge, ForeignKey

ERR_DELETE = "cannot delete {{column}} {{id}} because it still has edges in the Graph"


class Schema(object):
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


class Graph(Cluster):
    """
    A graph database consisting of a set of :class:`Table` s with :class:`U64` primary keys which serve as node IDs.

    Relationships are stored in 2D `Sparse` tensors whose coordinates take the form `[from_id, to_id]`.
    The two types of relationships are `Edge` and `ForeignKey`. These are distinguished by the graph schema--
    a table column which has an edge to itself is an `Edge`, otherwise it's a `ForeignKey`. `ForeignKey` relationships
    are automatically updated when a `Table` is updated, but `Edge` relationships require explicit management
    with the `add_edge` and `remove_edge` methods.
    """

    def _configure(self):
        schema = self._schema()

        if schema.tables:
            assert schema.chain is not None

        for (label, edge) in schema.edges.items():
            if edge.from_table == edge.to_table:
                setattr(self, label, schema.chain(Edge.zeros([I64.max(), I64.max()], Bool)))
            else:
                setattr(self, label, schema.chain(ForeignKey.zeros([I64.max(), I64.max()], Bool)))

        for name in schema.tables:
            if hasattr(self, name):
                raise ValueError(f"Graph already has an entry called {name}")

            setattr(self, name, schema.chain(graph_table(self, schema, name)))

    def _schema(self):
        return Schema(Sync)

    def add_edge(self, label, from_node, to_node):
        """Mark `from_node` -> `to_node` as `True` in the edge :class:`Tensor` with the given `label`."""

        edge = Sparse(Get(uri(self), label))
        return edge.write([from_node, to_node], True)

    def remove_edge(self, label, from_node, to_node):
        """Mark `from_node` -> `to_node` as `False` in the edge :class:`Tensor` with the given `label`."""

        edge = Sparse(Get(uri(self), label))
        return edge.write([from_node, to_node], False)


def graph_table(graph, schema, table_name):
    table_schema = schema.tables[table_name]

    if len(table_schema.key) != 1:
        raise ValueError("Graph table key must be a single column of type U64, not", table_schema.key)

    [key_col] = table_schema.key
    if key_col.dtype != U64:
        raise ValueError("Graph table key must be type U64, not", key_col.dtype)

    def delete_row(edge, adjacent, row):
        delete_from = adjacent.write([row[edge.column]], False)

        to_table = Table(URI(f"$self/{edge.to_table}"))
        if edge.cascade:
            delete_to = (
                to_table.delete({edge.column: row[edge.column]}),
                adjacent.write([slice(None), row[edge.column]], False))
        else:
            delete_to = If(
                adjacent[:, row[edge.column]].any(),
                BadRequest(ERR_DELETE, column=edge.column, id=row[edge.column]))

        if edge.from_table == table_name and edge.to_table == table_name:
            return delete_from, delete_to
        elif edge.from_table == table_name:
            return delete_from
        elif edge.to_table == table_name:
            return delete_to

    def maybe_update_row(edge, adjacent, row, cond, new_id):
        to_table = Table(URI(f"$self/{edge.to_table}"))

        args = closure(get_op(lambda row: [new_id, Tuple(row)[0]]))

        # assumes row[0] is always the key
        add = closure(put_op(lambda new_id, key: adjacent.write([new_id, key], True)))
        add = to_table.where({edge.column: new_id}).rows().map(args).for_each(add)

        return After(If(cond, delete_row(edge, adjacent, row)), add)

    class GraphTable(Table):
        def delete_row(self, key):
            row = self[key]
            deletes = []
            for label, edge in schema.edges.items():
                if table_name not in [edge.from_table, edge.to_table]:
                    continue

                adjacent = Sparse(uri(graph).append(label))
                deletes.append(delete_row(edge, adjacent, row))

            return If(row.is_some(), After(deletes, Table.delete_row(self, key)))

        def max_id(self):
            """Return the maximum ID present in this :class:`Table`."""

            row = Tuple(self.order_by([key_col.name], True).select([key_col.name]).rows().first())
            return U64(If(row.is_none(), 0, row[0]))

        def read_vector(self, node_ids):
            """Given a vector of `node_ids`, return a :class:`Stream` of :class:`Table` rows matching those IDs."""

            @closure
            @get_op
            def read_node(row: Tuple):
                return self[row[0]]

            return Sparse(uri(node_ids)).elements().map(read_node)

        def update_row(self, key, values):
            row = self[key]
            updates = []
            for label, edge in schema.edges.items():
                if table_name not in [edge.from_table, edge.to_table]:
                    continue

                if any(col.name == edge.column for col in table_schema.values):
                    adjacent = Sparse(uri(graph).append(label))
                    old_id = row[edge.column]
                    new_id = values[edge.column]
                    update = maybe_update_row(edge, adjacent, values.contains(edge.column), old_id, new_id)
                    updates.append(update)
                else:
                    # the edge is on the primary key, so it's not being updated
                    pass

            return After(Table.update_row(self, key, values), updates)

        def upsert(self, key, values):
            row = self[key]
            updates = []
            for label, edge in schema.edges.items():
                if table_name not in [edge.from_table, edge.to_table]:
                    continue

                adjacent = Sparse(uri(graph).append(label))

                if any(col.name == edge.column for col in table_schema.values):
                    value_index = [col.name for col in table_schema.values].index(edge.column)
                    new_id = values[value_index]
                else:
                    new_id = key[0]

                update = maybe_update_row(edge, adjacent, row, row.is_some(), new_id)
                updates.append(update)

            return After(Table.upsert(self, key, values), updates)

    return GraphTable(table_schema)
