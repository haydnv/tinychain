from tinychain.collection.tensor import einsum, Sparse
from tinychain.decorators import closure, get_op, post_op, put_op, delete_op
from tinychain.error import BadRequest
from tinychain.ref import After, Get, If, MethodSubject, While, With
from tinychain.state import Map, Tuple
from tinychain.value import Bool, Nil, I32, U64, String


ERR_DELETE = "cannot delete {{column}} {{id}} because it still has edges in the Graph"
DIM = I32.max_value()


class Schema(object):
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
        shape = node_ids.shape
        traversal = If(
            shape.eq([DIM]),
            While(cond, traverse, {"edge": self, "i": 0, "neighbors": node_ids}),
            BadRequest(String(f"an edge input vector has shape [{DIM}], not {{{shape}}}").render(shape=shape)))

        return Sparse.sub(Map(traversal)["neighbors"], node_ids)


class ForeignKey(Sparse):
    """A relationship between a primary key and a column in another `Table`."""

    def backward(self, node_ids):
        """Return a vector of primary node IDs, given a vector of foreign node IDs."""

        return einsum("ij,j->i", [self, node_ids])

    def forward(self, node_ids):
        """Return a vector of foreign node IDs, given a vector of primary node IDs."""

        return einsum("ij,i->j", [self, node_ids])
