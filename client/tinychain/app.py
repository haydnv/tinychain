from tinychain.chain import Sync
from tinychain.cluster import Cluster
from tinychain.collection.schema import Column, Graph as Schema
from tinychain.collection.table import Table
from tinychain.collection.tensor import Sparse
from tinychain.decorators import *
from tinychain.ref import Get
from tinychain.util import uri
from tinychain.value import String, Bool, I64, U64


class Graph(Cluster):
    def _configure(self):
        schema = self._schema()

        for label in schema.edges:
            setattr(self, f"edge_{label}", schema.chain(Sparse.zeros([I64.max(), I64.max()], Bool)))

        for name in schema.tables:
            if hasattr(self, name):
                raise ValueError(f"Graph already has an entry called f{name}")

            setattr(self, name, schema.chain(graph_table(schema.tables[name])))

    def _schema(self):
        return Schema(Sync)

    def add_edge(self, label, edge):
        (from_id, to_id) = edge
        edge = Sparse(Get(uri(self), String("edge_{{label}}").render(label=label)))
        return edge.write([from_id, to_id], True)

    def remove_edge(self, label, edge):
        (from_id, to_id) = edge
        edge = Sparse(Get(uri(self), String("edge_{{label}}").render(label=label)))
        return edge.write([from_id, to_id], False)


def graph_table(schema):
    if len(schema.key) != 1:
        raise ValueError("Graph table key must be a single column of type U64, not", schema.key)

    [key_col] = schema.key
    if key_col.dtype != U64:
        raise ValueError("Graph table key must be type U64, not", key_col.dtype)

    class GraphTable(Table):
        def max_id(self):
            return self.order_by(key_col.name, True).select([key_col.name]).rows().first()[0]

    return GraphTable(schema)
