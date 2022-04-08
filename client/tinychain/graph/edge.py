from ..collection.tensor import einsum, Sparse
from ..decorators import post
from ..error import BadRequest
from ..generic import Map
from ..scalar.number import Bool, I32, U32, U64
from ..scalar.ref import If, While
from ..scalar.value import String


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

    __spec__ = ((DIM, DIM), U32)

    def link(self, from_id, to_id):
        """Add a link from the :class:`Model` at `from_id` to the :class:`Model` at `to_id`"""

        return self[from_id, to_id].write(True)

    def unlink(self, from_id, to_id):
        """Remove the link from the :class:`Model` at `from_id` to the :class:`Model` at `to_id`, if present."""

        return self[from_id, to_id].write(False)

    def match(self, node_ids, degrees):
        """
        Traverse this `Edge` breadth-first from the given `node_ids`.

        Returns a vector view with the IDs of the matched nodes.
        Call `match` in a `While` loop to traverse multiple degrees of relationships.
        """

        @post
        def cond(i: U64):
            return i < degrees

        @post
        def traverse(edge: Sparse, i: U64, neighbors: Sparse):
            neighbors += (edge * neighbors).sum(1)
            return {"edge": edge, "i": i + 1, "neighbors": neighbors.copy()}

        node_ids = Sparse(node_ids)
        shape = node_ids.shape
        traversal = If(
            node_ids.ndim == 1,
            While(cond, traverse, {"edge": self, "i": 0, "neighbors": node_ids}),
            BadRequest(String(f"an edge input vector has shape [{DIM}], not {{{shape}}}").render(shape=shape)))

        return Sparse(Map(traversal)["neighbors"]) - node_ids


class ForeignKey(Sparse):
    """A relationship between a primary key and a column in another `Table`."""

    __spec__ = ((DIM, DIM), U32)

    def primary(self, node_ids):
        """Return a vector of primary node IDs, given a vector of foreign node IDs."""

        return einsum("ij,j->i", [self, node_ids])

    def foreign(self, node_ids):
        """Return a vector of foreign node IDs, given a vector of primary node IDs."""

        return einsum("ij,i->j", [self, node_ids])


class Vector(Sparse):
    """A `Vector` of node IDs used to query an :class:`Edge` or `ForeignKey`"""

    __spec__ = ((DIM,), Bool)

    @classmethod
    def create(cls):
        return cls(cls.__spec__)
