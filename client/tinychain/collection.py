"""Data structures responsible for storing a collection of :class:`Value`s"""

from .ref import OpRef
from .reflect import Meta
from .state import State
from .util import *
from .value import UInt, Nil


class Collection(State, metaclass=Meta):
    """
    Data structure responsible for storing a collection of :class:`Value`s.
    """

    __uri__ = uri(State) + "/collection"


class BTree(Collection):
    """
    A :class:`Chain` which keeps track of the entire update history of its subject.
    """

    __uri__ = uri(Collection) + "/btree"

    def count(self):
        """
        Return the number of keys in this BTree.

        To count the number of keys beginning with a specific prefix,
        call `btree.slice(prefix).count()`.
        """

        return UInt(OpRef.Get(uri(self) + "/count"))

    def delete(self):
        """
        Delete the contents of this BTree.
        
        To delete all keys beginning with a specific prefix, call
        `btree.slice(prefix).delete()`.
        """

        return Nil(OpRef.Delete(uri(self)))

    def insert(self, key):
        """
        Insert the given key into this BTree.

        If the key is already present, this is a no-op.
        """

        return Nil(OpRef.Put(uri(self) + "/insert", None, key))

    def slice(self, prefix):
        """Return a slice of this BTree in which all keys begin with the given prefix."""

        return BTree(OpRef.Get(uri(self) + "/slice", prefix))

