import unittest

from .test_btree import BTreeTests
from .test_einsum import EinsumTests
from .test_table import TableTests
from .test_tensor import DenseTests, SparseTests, TensorTests
from .test_value import ValueTests

unittest.main()
