from tinychain.chain import Sync
from tinychain.cluster import Cluster
from tinychain.collection.schema import Column, Graph as Schema
from tinychain.collection.table import Table
from tinychain.collection.tensor import einsum, Sparse
from tinychain.error import BadRequest
from tinychain.decorators import closure, get_op, post_op, put_op, delete_op
from tinychain.ref import After, Get, If, MethodSubject, While, With
from tinychain.state import Map, Tuple
from tinychain.util import uri, Context, URI
from tinychain.value import Bool, Nil, I64, U64, String


ERR_DELETE = "cannot delete {{column}} {{id}} because it still has edges in the Graph"


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


class Edge(Sparse):
    """A relationship between a primary key and itself."""

    def match(self, node_ids, degrees):
        """
        Traverse this `Edge` breadth-first from the given `node_ids` for the given number of `degrees`.

        Returns a new vector filled with the IDs of the matched nodes.
        """

        @post_op
        def cond(i: U64):
            return i < degrees

        @post_op
        def traverse(edge: Sparse, i: U64, neighbors: Sparse):
            neighbors += Sparse.sum(edge * neighbors, 1)
            return {"edge": edge, "i": i + 1, "neighbors": neighbors.copy()}

        node_ids = Sparse(node_ids)
        shape = node_ids.shape()
        traversal = If(
            shape.eq([I64.max()]),
            While(cond, traverse, {"edge": self, "i": 0, "neighbors": node_ids}),
            BadRequest(f"an edge input vector has shape [{I64.max()}], not {{shape}}", shape=shape))

        return Sparse.sub(Map(traversal)["neighbors"], node_ids)


class ForeignKey(Sparse):
    """A relationship between a primary key and a column in another `Table`."""

    def backward(self, node_ids):
        """Return a vector of primary node IDs, given a vector of foreign node IDs."""

        return einsum("ij,j->i", [self, node_ids])

    def forward(self, node_ids):
        """Return a vector of foreign node IDs, given a vector of primary node IDs."""

        return einsum("ij,i->j", [self, node_ids])


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
