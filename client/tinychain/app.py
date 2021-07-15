from tinychain.cluster import Cluster
from tinychain.collection.tensor import Sparse
from tinychain.value import Bool


class Graph(Cluster):
    def _configure(self):
        schema = self._schema()

        for name in schema.tables:
            if hasattr(self, name):
                raise ValueError(f"Graph already has an entry called f{name}")

            setattr(self, name, schema.tables[name])

        for name in schema.edges:
            tensor_name = f"edges_{name}"
            if hasattr(self, tensor_name):
                raise ValueError(f"Graph already has an entry called f{name}")

            setattr(self, tensor_name, Sparse(Bool, [0, 0]))

    def _schema(self):
        raise NotImplementedError("you must override the Graph._schema method")
