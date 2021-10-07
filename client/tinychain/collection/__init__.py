"""Data structures responsible for storing a collection of :class:`Value` s"""

from tinychain.util import to_json


class Column(object):
    """A column in the schema of a :class:`BTree` or :class:`Table`."""

    def __init__(self, name, dtype, max_size=None):
        self.name = name
        self.dtype = dtype
        self.max_size = max_size

    def __eq__(self, other):
        return self.name == other.name and self.dtype == other.dtype and self.max_size == other.max_size

    def __json__(self):
        if self.max_size is None:
            return to_json((self.name, self.dtype))
        else:
            return to_json((self.name, self.dtype, self.max_size))

    def __repr__(self):
        if self.max_size is None:
            return f"{self.name}: column type {self.dtype}"
        else:
            return f"{self.name}: column type {self.dtype}, max size {self.max_size}"
