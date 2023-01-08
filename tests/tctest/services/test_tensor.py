import itertools

import numpy as np
import tinychain as tc
import unittest

from ..process import DEFAULT_PORT
from .base import PersistenceTest


LEAD = f"http://127.0.0.1:{DEFAULT_PORT}"
NS = tc.URI("/test_tensor")


class TensorChainTests(PersistenceTest, unittest.TestCase):
    CACHE_SIZE = "100M"
    NUM_HOSTS = 4

    def service(self, chain_type):
        class Persistent(tc.service.Service):
            NAME = "tensor"
            VERSION = tc.Version("0.0.0")

            __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

            def __init__(self):
                schema = ([2, 3], tc.I32)
                self.dense = chain_type(tc.tensor.Dense(schema))
                self.sparse = chain_type(tc.tensor.Sparse(schema))

                tc.service.Service.__init__(self)

            @tc.put
            def overwrite(self, txn):
                txn.new = tc.tensor.Dense.constant([3], 2)
                return [
                    self.dense.write(txn.new),
                    self.sparse[0].write(txn.new)
                ]

            @tc.get
            def eq(self):
                return self.sparse == self.dense

        return Persistent()

    def execute(self, hosts):
        endpoints = {
            "tensor": tc.URI("/service/test_tensor/tensor/0.0.0")
        }

        endpoints["dense"] = endpoints["tensor"].append("dense")
        endpoints["sparse"] = endpoints["tensor"].append("sparse")

        hosts[0].put(endpoints["dense"], [0, 0], 1)
        hosts[1].put(endpoints["sparse"], [0, 0], 1)

        dense = expect_dense(tc.I32, [2, 3], [1, 0, 0, 0, 0, 0])
        sparse = expect_sparse(tc.I32, [2, 3], [[[0, 0], 1]])
        for host in hosts:
            actual = host.get(endpoints["dense"])
            self.assertEqual(actual, dense)

            actual = host.get(endpoints["sparse"])
            self.assertEqual(actual, sparse)

        hosts[1].stop()
        hosts[0].put(endpoints["tensor"].append("overwrite"))
        hosts[1].start()

        dense = expect_dense(tc.I32, [2, 3], [2] * 6)

        expected = np.zeros([2, 3])
        expected[0] = (np.ones([3]) * 2)
        sparse = expect_sparse(tc.I32, [2, 3], expected)

        eq = expect_dense(tc.Bool, [2, 3], [True, True, True, False, False, False])

        for host in hosts:
            actual = host.get(endpoints["dense"])
            self.assertEqual(actual, dense)

            actual = host.get(endpoints["sparse"])
            self.assertEqual(actual, sparse)

            actual = host.get(endpoints["tensor"].append("eq"))
            self.assertEqual(actual, eq)


def expect_dense(dtype, shape, flat):
    return {
        str(tc.URI(tc.tensor.Dense)): [
            [list(shape), str(tc.URI(dtype))],
            list(flat),
        ]
    }


def expect_sparse(dtype, shape, values):
    if isinstance(values, np.ndarray):
        values = nparray_to_sparse(values, dtype)

    return {
        str(tc.URI(tc.tensor.Sparse)): [
            [list(shape), str(tc.URI(dtype))],
            list(values),
        ]
    }


def nparray_to_sparse(arr, dtype):
    dtype = float if issubclass(dtype, tc.Float) else int
    zero = dtype(0)
    coords = itertools.product(*[range(dim) for dim in arr.shape])
    sparse = [[list(coord), n] for (coord, n) in zip(coords, (dtype(n) for n in arr.flatten())) if n != zero]
    return sparse
