from tinychain.chain import Sync
from tinychain.cluster import Cluster
from tinychain.collection.schema import Column, Graph as Schema, Table as TableSchema
from tinychain.collection.table import Table
from tinychain.collection.tensor import Sparse
from tinychain.ref import After, If
from tinychain.value import Bool, I64, U64


NODE_ID = "node_id"


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

            [key] = [col for col in schema.tables[table_name].columns() if col.name == column_name]
            setattr(self, f"node_{table_name}_{column_name}", schema.chain(node_table(key)))

        for name in schema.edges:
            setattr(self, f"edges_{name}", schema.chain(Sparse.zeros([I64.max(), I64.max()], Bool)))

    def _schema(self):
        return Schema(Sync)


def graph_table(graph_schema, table_name):
    edge_source = {}
    edge_dest = {}

    for edge_name in graph_schema.edges:
        from_node, to_node = graph_schema.edges[edge_name]
        prefix = f"{table_name}."

        if from_node.startswith(prefix):
            edge_source[edge_name] = from_node[len(prefix):]

        if to_node.startswith(prefix):
            edge_dest[edge_name] = to_node[len(prefix):]

    class GraphTable(Table):
        def delete_row(self, key):
            return self._delete("", key)

        def update_row(self, key, values):
            return self._put("", key, values)

        def upsert(self, key, values):
            return self._put("", key, values)

    return GraphTable(graph_schema.tables[table_name])


def node_table(key_column):
    class NodeTable(Table):
        def create_id(self, key):
            new_id = self.max_id()
            return After(self.insert([key, new_id]), new_id)

        def get_id(self, key):
            return self[(key,)][NODE_ID]

        def get_or_create_id(self, key):
            return If(self.contains([key]), self.get_id(key), self.create_id(key))

        def max_id(self):
            row = self.order_by([NODE_ID], True).select([NODE_ID]).first()
            return row[0]

    schema = TableSchema([key_column], [Column(NODE_ID, U64)])
    return NodeTable(schema.create_index(NODE_ID, [NODE_ID]))
