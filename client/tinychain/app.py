from tinychain.chain import Sync
from tinychain.cluster import Cluster
from tinychain.collection.schema import Column, Graph as Schema
from tinychain.collection.table import Table
from tinychain.collection.tensor import Sparse
from tinychain.error import BadRequest
from tinychain.decorators import delete_op
from tinychain.ref import After, Get, If
from tinychain.state import Tuple
from tinychain.util import uri
from tinychain.value import String, Bool, I64, U64


class Graph(Cluster):
    def _configure(self):
        schema = self._schema()

        for label in schema.edges:
            setattr(self, f"edge_{label}", schema.chain(Sparse.zeros([I64.max(), I64.max()], Bool)))

        for name in schema.tables:
            if hasattr(self, name):
                raise ValueError(f"Graph already has an entry called {name}")

            setattr(self, name, schema.chain(graph_table(self, schema, name)))

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


def graph_table(graph, schema, table_name):
    table_schema = schema.tables[table_name]

    if len(table_schema.key) != 1:
        raise ValueError("Graph table key must be a single column of type U64, not", table_schema.key)

    [key_col] = table_schema.key
    if key_col.dtype != U64:
        raise ValueError("Graph table key must be type U64, not", key_col.dtype)

    class GraphTable(Table):
        def delete_row(self, key):
            row = self[key]
            deletes = []
            for label, edge in schema.edges.items():
                adjacent = Sparse(uri(graph).append(f"edge_{label}"))
    
                if edge.to_table == edge.from_table:
                    delete = adjacent.write([slice(None), row[edge.column]], False)
                    deletes.append(delete)
                elif edge.from_table == table_name:
                    delete = adjacent.write([row[edge.column]], False)

                    if edge.cascade:
                        to_table = getattr(graph, edge.to_table)
                        cascade = delete_op(lambda to: to_table.delete({edge.column: to[0]}))
                        cascade = adjacent.filled([row[edge.column]]).rows().for_each(cascade)
                        delete = After(cascade, delete)
                    else:
                        err = String("cannot delete from {{table}}: Graph still has edges to {{column}}").render(
                            table=table_name, column=edge.column)

                        delete = If(adjacent[row[edge.column]].any(), BadRequest(err))

                    deletes.append(delete)
                elif edge.to_table == table_name:
                    delete = adjacent.write([row[edge.column], row[edge.column]], False)
                    delete = If(adjacent[:, row[edge.column]].cast(U64).sum() == 1, delete)
                    deletes.append(delete)

            return After(deletes, Table.delete_row(self, key))

        def update_row(self, key, values):
            row = self[key]
            updates = []
            for label, edge in schema.edges.items():
                if table_name not in [edge.from_table, edge.to_table]:
                    continue

                if any(col.name == edge.column for col in table_schema.values):
                    adjacent = Sparse(uri(graph).append(f"edge_{label}"))
                    old_id = row[edge.column]
                    new_id = values[edge.column]
                    update = self._maybe_update_row(edge, adjacent, values.contains(edge.column), old_id, new_id)
                    updates.append(update)

            return After(updates, Table.update_row(self, key, values))

        def upsert(self, key, values):
            row = self[key]
            updates = []
            for label, edge in schema.edges.items():
                if table_name not in [edge.from_table, edge.to_table]:
                    continue

                adjacent = Sparse(uri(graph).append(f"edge_{label}"))

                if any(col.name == edge.column for col in table_schema.values):
                    old_id = row[edge.column]
                    value_index = [col.name for col in table_schema.values].index(edge.column)
                    new_id = values[value_index]
                    update = self._maybe_update_row(edge, adjacent, row.is_some(), old_id, new_id)
                    updates.append(update)
                else:
                    # the edge column can't be updated, so don't worry about invalidating old edges
                    if edge.from_table == table_name and edge.to_table == table_name:
                        # this is not a foreign key, so the developer has to add edges explicitly
                        pass
                    else:
                        key_index = [col.name for col in table_schema.key].index(edge.column)
                        update = adjacent.write([key[key_index], key[key_index]], True)
                        updates.append(update)

            return After(updates, Table.upsert(self, key, values))

        def _maybe_update_row(self, edge, adjacent, update_cond, old_id, new_id):
            if edge.from_table == edge.to_table:
                add = None  # this is not a foreign key, so the developer has to add edges explicitly

                if edge.cascade:
                    remove = (adjacent.write([old_id], False),
                              adjacent.write([slice(None), new_id], False))
                else:
                    err = String("cannot delete from {{table}}: Graph still has edges to {{column}}").render(
                        table=edge.from_table, column=edge.column)

                    still_exists = adjacent[old_id].any().logical_or(adjacent[slice(None), old_id].any())
                    remove = If(still_exists, BadRequest(err))
            elif table_name == edge.from_table:
                remove = (adjacent.write([old_id], False), adjacent.write([slice(None), old_id], False))
                add = adjacent.write([new_id, new_id], True)
            elif table_name == edge.to_table:
                remove = If(adjacent[:, old_id].cast(U64).sum() == 1, adjacent.write([old_id, old_id], False))
                add = adjacent.write([new_id, new_id], True)

            remove = If(update_cond, remove)

            if add is None:
                return remove
            else:
                return After(remove, add)

        def max_id(self):
            row = Tuple(self.order_by([key_col.name], True).select([key_col.name]).rows().first())
            return U64(If(row.is_none(), 0, row[0]))

    return GraphTable(table_schema)
