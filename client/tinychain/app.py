from tinychain.chain import Sync
from tinychain.cluster import Cluster
from tinychain.collection.schema import Graph as Schema
from tinychain.collection.table import Table
from tinychain.collection.tensor import Sparse
from tinychain.value import Bool


class Graph(Cluster):
    def _configure(self):
        schema = self._schema()

        for name in schema.tables:
            if hasattr(self, name):
                raise ValueError(f"Graph already has an entry called f{name}")

            setattr(self, name, schema.chain(Table(schema.tables[name])))

        for name in schema.edges:
            (from_node, to_node) = schema.edges[name]

            if '.' not in from_node:
                raise ValueError(f"{from_node} must specify a column")

            if '.' not in to_node:
                raise ValueError(f"{to_node} must specify a column")

            tensor_name = f"edges_{name}"
            if hasattr(self, tensor_name):
                raise ValueError(f"Graph already has an entry called f{name}")

            setattr(self, tensor_name, schema.chain(Sparse.zeros([0, 0], Bool)))

    def _schema(self):
        return Schema(Sync)
