from tinychain.chain import Sync
from tinychain.cluster import Cluster
from tinychain.collection.schema import Column, Graph as Schema, Table as TableSchema
from tinychain.collection.table import Table
from tinychain.collection.tensor import Sparse
from tinychain.value import Bool, I64, U64


class Graph(Cluster):
    def _configure(self):
        schema = self._schema()

        for name in schema.tables:
            if hasattr(self, name):
                raise ValueError(f"Graph already has an entry called f{name}")

            setattr(self, name, schema.chain(graph_table(schema.tables[name])))

        nodes = set()
        for from_node, to_node in schema.edges.values():
            nodes.add(from_node)
            nodes.add(to_node)

        for node in nodes:
            if '.' in node:
                table_name, column_name = node.split('.')
            else:
                raise ValueError(f"{node} must specify a column")

            key = [col for col in schema.tables[table_name].columns() if col.name == column_name]
            node_schema = TableSchema(key, [Column("node_id", U64)])
            setattr(self, f"node_{table_name}_{column_name}", schema.chain(NodeTable(node_schema)))

        for name in schema.edges:
            setattr(self, f"edges_{name}", schema.chain(Sparse.zeros([I64.max(), I64.max()], Bool)))

    def _schema(self):
        return Schema(Sync)


class NodeTable(Table):
    def max_id(self):
        def max(acc):
            max_id, node_id = acc
            return node_id if node_id > max_id else max_id

        return self.keys(["node_id"]).fold(0, max)


def graph_table(table_schema):
    class GraphTable(Table):
        def delete_row(self, key):
            return self._delete("", key)

        def update_row(self, key, values):
            return self._put("", key, values)

        def upsert(self, key, values):
            return self._put("", key, values)

    return GraphTable(table_schema)
