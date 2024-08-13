import itertools
import numpy as np
import tinychain as tc
import unittest

from .base import PersistenceTest
from ..process import DEFAULT_PORT


LEAD = f"http://127.0.0.1:{DEFAULT_PORT}"
NS = tc.URI("/test_tensor")


class TensorChainTests(PersistenceTest, unittest.TestCase):
    CACHE_SIZE = "1G"
    NUM_HOSTS = 3

    def service(self, chain_type):
        class Persistent(tc.service.Service):
            NAME = "tensor"
            VERSION = tc.Version("0.0.0")

            __uri__ = tc.service.service_uri(LEAD, NS, NAME, VERSION)

            def __init__(self):
                schema = (tc.I32, [2, 3])
                self.dense = chain_type(tc.tensor.Dense(schema))
                self.sparse = chain_type(tc.tensor.Sparse(schema))
                tc.service.Service.__init__(self)

            @tc.put
            def overwrite(self, txn):
                txn.new = tc.tensor.Dense.constant([3], 2)
                return self.dense.write(txn.new)

            @tc.get
            def eq(self):
                return self.dense == self.sparse

        return Persistent()

    def execute(self, hosts):
        endpoints = {"tensor": tc.URI("/service/test_tensor/tensor/0.0.0")}

        endpoints["dense"] = endpoints["tensor"].append("dense")
        endpoints["sparse"] = endpoints["tensor"].append("sparse")

        hosts[0].put(endpoints["dense"], [0, 0], 1)
        hosts[-1].put(endpoints["sparse"], [0, 0], 2)

        dense = expect_dense(tc.I32, [2, 3], [1, 0, 0, 0, 0, 0])
        sparse = expect_sparse(tc.I32, [2, 3], [[[0, 0], 2]])

        for host in hosts:
            actual = host.get(endpoints["dense"])
            self.assertEqual(actual, dense)

            actual = host.get(endpoints["sparse"])
            self.assertEqual(actual, sparse)

        hosts[1].stop()
        hosts[2].put(endpoints["tensor"].append("overwrite"))
        hosts[1].start(wait_time=10)

        dense = expect_dense(tc.I32, [2, 3], [2] * 6)

        for host in hosts:
            actual = host.get(endpoints["dense"])
            self.assertEqual(actual, dense, host)

            actual = host.get(endpoints["sparse"])
            self.assertEqual(actual, sparse, host)

        eq = expect_dense(tc.Bool, [2, 3], [True] + [False] * 5)

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
            [str(tc.URI(dtype)), list(shape)],
            list(flat),
        ]
    }


def expect_sparse(dtype, shape, values):
    if isinstance(values, np.ndarray):
        values = nparray_to_sparse(values, dtype)

    return {
        str(tc.URI(tc.tensor.Sparse)): [
            [str(tc.URI(dtype)), list(shape)],
            list(values),
        ]
    }


def nparray_to_sparse(arr, dtype):
    dtype = float if issubclass(dtype, tc.Float) else int
    zero = dtype(0)
    coords = itertools.product(*[range(dim) for dim in arr.shape])
    sparse = [
        [list(coord), n]
        for (coord, n) in zip(coords, (dtype(n) for n in arr.flatten()))
        if n != zero
    ]
    return sparse


def printlines(n):
    for _ in range(n):
        print()
