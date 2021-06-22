import itertools
import numpy as np
import tinychain as tc
import unittest

from testutils import PORT, start_host, PersistenceTest


ENDPOINT = "/transact/hypothetical"


class DenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_dense_tensor")

    def testConstant(self):
        c = 1.414
        shape = [3, 2, 1]

        cxt = tc.Context()
        cxt.tensor = tc.tensor.Dense.constant(shape, c)
        cxt.result = tc.After(cxt.tensor.write([0, 0, 0], 0), cxt.tensor)

        expected = expect_dense(tc.F64, shape, [0] + [c] * (product(shape) - 1))
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expected, actual)

    def testSlice(self):
        shape = [2, 5]

        cxt = tc.Context()
        cxt.tensor = tc.tensor.Dense.arange(shape, 1, 11)
        cxt.result = cxt.tensor[1, 2:-1]

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.I64, [2], np.arange(1, 11).reshape([2, 5])[1, 2:-1])
        self.assertEqual(actual, expected)

    def testAdd(self):
        shape = [5, 2, 1]

        cxt = tc.Context()
        cxt.left = tc.tensor.Dense.arange(shape, 1., 6.)
        cxt.right = tc.tensor.Dense.constant([5], 2)
        cxt.result = cxt.left + cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.F64, shape, np.arange(1, 6, 0.5) + 2)
        self.assertEqual(actual, expected)

    def testDiv(self):
        shape = [3]

        cxt = tc.Context()
        cxt.left = tc.tensor.Dense.arange(shape, 2., 8.)
        cxt.right = tc.tensor.Dense.constant([1], 2)
        cxt.result = cxt.left / cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.F64, shape, np.arange(1, 4))
        self.assertEqual(actual, expected)

    def testMul(self):
        shape = [5, 2, 1]

        cxt = tc.Context()
        cxt.left = tc.tensor.Dense.arange(shape, 1, 11)
        cxt.right = tc.tensor.Dense.constant([5], 2)
        cxt.result = cxt.left * cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.I64, shape, np.arange(1, 11) * 2)
        self.assertEqual(actual, expected)

    def testSub(self):
        shape = [1, 3]

        cxt = tc.Context()
        cxt.left = tc.tensor.Dense.arange(shape, 0, 6)
        cxt.right = tc.tensor.Dense.constant([1], 2)
        cxt.result = cxt.left - cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.I64, shape, np.arange(-2, 4, 2))
        self.assertEqual(actual, expected)

    def testLogic(self):
        big = [20, 20, 10]
        trailing = [10]

        cxt = tc.Context()
        cxt.big_ones = tc.tensor.Dense.ones(big, tc.U8)
        cxt.big_zeros = tc.tensor.Dense.zeros(big, tc.U8)
        cxt.true = tc.tensor.Dense.ones(trailing)
        cxt.false = tc.tensor.Dense.zeros(trailing)
        cxt.result = [
            cxt.big_ones.logical_and(cxt.false).any(),
            cxt.big_ones.logical_and(cxt.true).all(),
            cxt.big_zeros.logical_or(cxt.true).all(),
            cxt.big_zeros.logical_or(cxt.false).any(),
            cxt.big_ones.logical_xor(cxt.big_zeros).all(),
        ]

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, [False, True, True, False, True])

    def testProduct(self):
        shape = [2, 3, 4]
        axis = 1

        cxt = tc.Context()
        cxt.big = tc.tensor.Dense.arange(shape, 0, 24)
        cxt.result = cxt.big.product(axis)

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.product(np.arange(0, 24).reshape(shape), axis)
        self.assertEqual(actual, expect_dense(tc.I64, [2, 4], expected.flatten()))

    def testProductAll(self):
        shape = [2, 3]

        cxt = tc.Context()
        cxt.big = tc.tensor.Dense.arange(shape, 1, 7)
        cxt.result = cxt.big.product()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, product(range(1, 7)))

    def testSum(self):
        shape = [4, 2, 3, 5]
        axis = 2

        cxt = tc.Context()
        cxt.big = tc.tensor.Dense.arange(shape, 0., 120.)
        cxt.result = cxt.big.sum(axis)

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.sum(np.arange(0, 120).reshape(shape), axis)
        self.assertEqual(actual, expect_dense(tc.F64, [4, 2, 5], expected.flatten()))

    def testSumAll(self):
        shape = [5, 2]

        cxt = tc.Context()
        cxt.big = tc.tensor.Dense.arange(shape, 0, 10)
        cxt.result = cxt.big.sum()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, sum(range(10)))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


class SparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_sparse_tensor")

    def testCreate(self):
        shape = [2, 5]
        coord = [0, 0]
        value = 1

        cxt = tc.Context()
        cxt.tensor = tc.tensor.Sparse.zeros(shape, tc.I32)
        cxt.result = tc.After(cxt.tensor.write(coord, value), cxt.tensor)

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_sparse(tc.I32, shape, [[coord, value]])
        self.assertEqual(actual, expected)

    def testWriteAndSlice(self):
        shape = [2, 5]

        cxt = tc.Context()
        cxt.tensor = tc.tensor.Sparse.zeros(shape)
        cxt.result = tc.After(cxt.tensor.write([None, slice(2, -1)], 1), cxt.tensor)

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_sparse(tc.F32, shape, [[[0, 2], 1], [[0, 3], 1], [[1, 2], 1], [[1, 3], 1]])
        self.assertEqual(actual, expected)

    def testAdd(self):
        shape = [5, 2, 3]

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape)
        cxt.small = tc.tensor.Sparse.zeros([3])
        cxt.result = tc.After([
            cxt.big.write([1], 1),
            cxt.small.write([1], 2),
        ], cxt.big + cxt.small)

        actual = self.host.post(ENDPOINT, cxt)

        big = np.zeros(shape)
        big[1] = 1
        small = np.zeros([3])
        small[1] = 2
        expected = big + small

        expected = expect_sparse(tc.F32, shape, expected)
        self.assertEqual(actual, expected)

    def testDiv(self):
        shape = [3, 2, 4]

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape)
        cxt.small = tc.tensor.Sparse.zeros([1, 1])
        cxt.result = tc.After([
            cxt.big.write([slice(2)], 1),
            cxt.small.write([0], -2),
        ], cxt.big / cxt.small)

        actual = self.host.post(ENDPOINT, cxt)

        big = np.zeros(shape)
        big[:2] = 1.
        small = np.zeros([1, 1])
        small[0] = -2.
        expected = big / small

        expected = expect_sparse(tc.F32, shape, expected)
        self.assertEqual(actual, expected)

    def testMul(self):
        shape = [3, 5, 2]

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape)
        cxt.small = tc.tensor.Sparse.zeros([5, 2])
        cxt.result = tc.After([
            cxt.big.write([None, slice(1, -2)], 2),
            cxt.small.write([1], 3),
        ], cxt.big * cxt.small)

        actual = self.host.post(ENDPOINT, cxt)

        big = np.zeros(shape)
        big[:, 1:-2] = 2
        small = np.zeros([5, 2])
        small[1] = 3
        expected = big * small

        expected = expect_sparse(tc.F32, shape, expected)
        self.assertEqual(actual, expected)

    def testSub(self):
        shape = [3, 5, 2]

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape, tc.I16)
        cxt.small = tc.tensor.Sparse.zeros([5, 2], tc.U32)
        cxt.result = tc.After([
            cxt.big.write([None, slice(1, -2)], 2),
            cxt.small.write([1], 3),
        ], cxt.small - cxt.big)

        actual = self.host.post(ENDPOINT, cxt)

        big = np.zeros(shape)
        big[:, 1:-2] = 2
        small = np.zeros([5, 2])
        small[1] = 3
        expected = small - big

        expected = expect_sparse(tc.I16, shape, expected)
        self.assertEqual(actual, expected)

    def testSum(self):
        shape = [2, 4, 3, 5]
        axis = 1

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape, tc.I32)
        cxt.result = tc.After(cxt.big.write([0, slice(1, 3)], 2), cxt.big.sum(axis))

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.zeros(shape, dtype=np.int32)
        expected[0, 1:3] = 2
        expected = expected.sum(axis)
        self.assertEqual(actual, expect_sparse(tc.I32, [2, 3, 5], expected))

    def testProduct(self):
        self.maxDiff = None
        shape = [2, 4, 3, 5]
        axis = 2

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape, tc.I32)
        cxt.result = tc.After(cxt.big.write([0, slice(1, 3)], 2), cxt.big.product(axis))

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.zeros(shape, dtype=np.int32)
        expected[0, 1:3] = 2
        expected = expected.prod(axis)
        self.assertEqual(actual, expect_sparse(tc.I32, [2, 4, 5], expected))


class ChainTests(PersistenceTest, unittest.TestCase):
    CACHE_SIZE = "10M"
    NUM_HOSTS = 4
    NAME = "tensor"

    def cluster(self, chain_type):
        class Persistent(tc.Cluster):
            __uri__ = tc.URI(f"http://127.0.0.1:{PORT}/test/tensor")

            def _configure(self):
                schema = tc.schema.Tensor([2, 3], tc.I32)
                self.dense = chain_type(tc.tensor.Dense(schema))

            @tc.put_method
            def overwrite(self, txn):
                txn.new = tc.tensor.Dense.constant([3], 2)
                return self.dense.write(None, txn.new)

        return Persistent

    def execute(self, hosts):
        hosts[0].put("/test/tensor/dense", [0, 0], 1)

        for host in hosts:
            actual = host.get("/test/tensor/dense")
            expected = expect_dense(tc.I32, [2, 3], [1, 0, 0, 0, 0, 0])
            self.assertEqual(actual, expected)

        hosts[0].put("/test/tensor/overwrite")

        expected = expect_dense(tc.I32, [2, 3], [2] * 6)
        for host in hosts:
            actual = host.get("/test/tensor/dense")
            self.assertEqual(actual, expected)


def expect_dense(dtype, shape, flat):
    return {
        str(tc.uri(tc.tensor.Dense)): [
            [shape, str(tc.uri(dtype))],
            list(flat),
        ]
    }


def expect_sparse(dtype, shape, values):
    if isinstance(values, np.ndarray):
        values = nparray_to_sparse(values, dtype)

    return {
        str(tc.uri(tc.tensor.Sparse)): [
            [shape, str(tc.uri(dtype))],
            list(values),
        ]
    }


def product(seq):
    p = 1
    for n in seq:
        p *= n

    return p


def nparray_to_sparse(arr, dtype):
    dtype = float if issubclass(dtype, tc.Float) else int
    zero = dtype(0)
    coords = itertools.product(*[range(dim) for dim in arr.shape])
    sparse = [[list(coord), n] for (coord, n) in zip(coords, (dtype(n) for n in arr.flatten())) if n != zero]
    return sparse


if __name__ == "__main__":
    unittest.main()
