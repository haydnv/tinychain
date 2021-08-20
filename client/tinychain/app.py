from tinychain.chain import Sync
from tinychain.cluster import Cluster
from tinychain.collection.schema import Column, Graph as Schema, Table as TableSchema
from tinychain.collection.table import Table
from tinychain.collection.tensor import Sparse
from tinychain.ref import After, Before, If
from tinychain.state import Map, Tuple
from tinychain.util import uri
from tinychain.value import Bool, I64, U64


NODE_ID = "node_id"


class Graph(Cluster):
    def _configure(self):
        schema = self._schema()

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
            setattr(self, f"edge_{name}", schema.chain(Sparse.zeros([I64.max(), I64.max()], Bool)))

        for name in schema.tables:
            if hasattr(self, name):
                raise ValueError(f"Graph already has an entry called f{name}")

            setattr(self, name, schema.chain(graph_table(self, schema, name)))

    def _schema(self):
        return Schema(Sync)


def graph_table(graph, schema, table_name):
    nodes = {}
    prefix = f"{table_name}."
    for edge_name in schema.edges:
        from_node, to_node = schema.edges[edge_name]

        if from_node.startswith(prefix):
            nodes[from_node[len(prefix):]] = reference_node(graph, from_node), reference_node(graph, to_node)

    table_schema = schema.tables[table_name]
    key_columns = [col.name for col in table_schema.key]
    value_columns = [col.name for col in table_schema.values]

    class GraphTable(Table):
        def delete_row(self, key):
            # for each edge whose source is this table
            #   if this row's value is unique, delete the node from the node table
            # finally, delete the row and the edge

            row = self[key]

            deletes = []
            for col_name in nodes:
                from_node, to_node = nodes[col_name]
                from_node_id = If(
                    self.count({col_name: row[col_name]}) == 1,
                    from_node.remove(row[col_name]),
                    from_node.get_id(row[col_name]))

                deletes.append(from_node_id)

            return After(If(row.is_none(), None, deletes), Table.delete_row(key))

        def update_row(self, key, values):
            # for each edge whose source is this table
            #   if the new value is different
            #       if this row's value is unique, delete the node from the node table
            #       insert the new value into the node table
            #       delete the old edge and insert the new edge

            row = Map(Before(Table.update_row(self, key, values), self[key]))

            updates = []
            for col_name in nodes:
                from_node, to_node = nodes[col_name]
                old_value = row[col_name]
                old_node_id = If(
                    self.count({col_name: old_value}) == 1,
                    from_node.remove(old_value),
                    from_node.get_id(old_value))

                if col_name in value_columns:
                    new_node_id = from_node.add(values[col_name])
                    update = If(values.contains(col_name), (new_node_id, old_node_id))
                else:
                    new_value = key[key_columns.index(col_name)]
                    new_node_id = from_node.add(new_value)
                    update = If(old_value != new_value, (new_node_id, old_node_id))

                updates.append(update)

            return updates

        def upsert(self, key, values):
            # for each edge whose source is this table
            #   if the new value is different
            #       if this row's value is unique, delete the node from the node table
            #   delete the old edge
            #   insert the new value into the node table
            #   insert the new edge
            # finally, upsert the row

            row = self[key]

            creates = []
            deletes = []
            for col_name in nodes:
                if col_name in key_columns:
                    new_value = key[key_columns.index(col_name)]
                else:
                    new_value = values[value_columns.index(col_name)]

                from_node, to_node = nodes[col_name]

                old_value = row[col_name]
                old_from_node_id = If(
                    self.count({col_name: old_value}) == 1,
                    from_node.remove(old_value),
                    from_node.get_id(old_value))

                deletes.append(If(old_value != new_value, old_from_node_id))

                new_from_node_id = from_node.add(new_value)
                creates.append(new_from_node_id)

            return After(If(row.is_none(), creates, None), (creates, Table.upsert(self, key, values)))

    return GraphTable(table_schema)


def node_table(key_column):
    class NodeTable(Table):
        def add(self, key):
            return If(self.contains([key]), self.get_id(key), self.create_id(key))

        def create_id(self, key):
            new_id = self.max_id()
            return After(self.insert([key], [new_id]), new_id)

        def get_id(self, key):
            return self[(key,)][NODE_ID]

        def max_id(self):
            row = self.order_by([NODE_ID], True).select([NODE_ID]).rows().first()
            return If(row.is_none(), 0, Tuple(row)[0])

        def remove(self, key):
            return Before(self.delete_row([key]), self.get_id(key))

    schema = TableSchema([key_column], [Column(NODE_ID, U64)])
    return NodeTable(schema.create_index(NODE_ID, [NODE_ID]))


def reference_edge(graph, edge_name):
    tensor_name = f"edge_{edge_name}"
    edge_class = type(getattr(graph, tensor_name))
    return edge_class(uri(graph).append(tensor_name))


def reference_node(graph, node):
    assert '.' in node
    table_name, col_name = node.split('.')
    node_name = f"node_{table_name}_{col_name}"
    table_class = type(getattr(graph, node_name))
    return table_class(uri(graph).append(node_name))
