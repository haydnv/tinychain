from tinychain.ref import Post, Put
from tinychain.reflect import is_ref
from tinychain.state import Map, State, Tuple
from tinychain.util import uri


class Collection(State):
    """Data structure responsible for storing a collection of :class:`Value`s."""

    __uri__ = uri(State) + "/collection"

    @classmethod
    def copy_from(cls, schema, source):
        """
        Copy a :class:`Collection` from a :class:`Stream` of values.

        Example:
            .. highlight:: python
            .. code-block:: python

                columns = [tc.Column("first", tc.String, 128), tc.Column("second", tc.String, 128)]
                row_schema = tc.schema.BTree(*columns)
                table_schema = tc.schema.Table([columns[0]], [columns[1]])

                btree = tc.btree.load(row_schema, [["hello", "world"]])
                table = tc.table.copy_from(table_schema, btree.keys())
        """

        return cls(Post(uri(cls) + "/copy_from", Map(schema=schema, source=source)))

    @classmethod
    def load(cls, schema, data):
        """
        Load a :class:`Collection` from an external data set.

        Example:
            .. highlight:: python
            .. code-block:: python

                elements = range(10)
                dense = tc.tensor.Dense([2, 5], tc.I32, elements)
        """

        if is_ref(schema):
            raise ValueError(f"cannot load schema {schema} (consider calling `Collection.copy_from` instead)")

        if is_ref(data):
            raise ValueError(f"cannot load data {data} (consider calling `Collection.copy_from` instead)")

        return cls(Put(cls, schema, data))

    def copy(self):
        """Return a copy of this `Collection`."""

        return self.__class__.copy_from(self.schema(), self)

    def schema(self):
        """Return the schema of this `Collection`."""

        return self._get("schema", rtype=Tuple)
