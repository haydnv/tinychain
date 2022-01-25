import functools
import itertools
import math
import operator

import numpy as np
import tinychain as tc
import unittest

from testutils import DEFAULT_PORT, start_host, PersistenceTest


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
        cxt.result = tc.After(cxt.tensor[0, 0, 0].write(0), cxt.tensor)

        expected = expect_dense(tc.F64, shape, [0] + [c] * (np.product(shape) - 1))
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

    def testAssignSlice(self):
        cxt = tc.Context()
        cxt.big = tc.tensor.Dense.zeros([2, 2, 5], tc.I32)
        cxt.small = tc.tensor.Dense.arange([2, 5], 1, 11)
        cxt.result = tc.After(cxt.big[1, :2].write(cxt.small[0]), cxt.big)

        actual = self.host.post(ENDPOINT, cxt)

        expected = np.zeros([2, 2, 5], np.int64)
        expected[1, 0:2] = np.arange(1, 11).reshape(2, 5)[0]
        expected = expect_dense(tc.I32, [2, 2, 5], expected.flatten())

        self.assertEqual(actual, expected)

    def testAdd(self):
        cxt = tc.Context()
        cxt.left = tc.tensor.Dense.arange([5, 2, 2], 1., 21.)
        cxt.right = tc.tensor.Dense.constant([2, 5, 1, 2], 2)
        cxt.result = cxt.left + cxt.right

        actual = self.host.post(ENDPOINT, cxt)

        left = np.arange(1., 21., 1.).reshape([5, 2, 2])
        right = np.ones([2, 5, 1, 2], np.int32) * 2
        expected = expect_dense(tc.F64, [2, 5, 2, 2], (left + right).flatten())

        self.assertEqual(actual, expected)

    def testCast(self):
        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.random_normal([3, 3])
        cxt.y1 = cxt.x / tc.Float(3.)
        cxt.y2 = cxt.x / tc.Int(3).cast(tc.Float)
        cxt.y3 = cxt.x.cast(tc.F32) / 3
        cxt.result = (cxt.y1.dtype, cxt.y2.dtype, cxt.y3.dtype)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, [{tc.uri(tc.Class): {tc.uri(tc.F64): {}}}] * 3)

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

        left = np.arange(1, 11).reshape(shape)
        right = np.ones([5]) * 2
        expected = left * right
        expected = expect_dense(tc.I64, list(expected.shape), expected.flatten())
        self.assertEqual(actual, expected)

    def testMulWithBroadcast(self):
        tau = np.array([[4.188]])
        v = np.array([[1], [0.618]])

        cxt = tc.Context()
        cxt.tau = load_dense(tau)
        cxt.v = load_dense(v)
        cxt.result = cxt.tau * tc.tensor.einsum("ij,kj->ik", [cxt.v, cxt.v])

        actual = self.host.post(ENDPOINT, cxt)
        expected = tau * (v @ v.T)
        self.assertEqual(expected.shape, tuple(actual[tc.uri(tc.tensor.Dense)][0][0]))
        self.assertTrue(np.allclose(expected.flatten(), actual[tc.uri(tc.tensor.Dense)][1]))

    def testSub(self):
        shape = [1, 3]

        cxt = tc.Context()
        cxt.left = tc.tensor.Dense.arange(shape, 0, 6)
        cxt.right = tc.tensor.Dense.constant([1], 2)
        cxt.result = cxt.left - cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.I64, shape, np.arange(-2, 4, 2))
        self.assertEqual(actual, expected)

    def testSplit(self):
        splits = 3
        shape = (6, 30)
        x = np.ones(shape, dtype=np.int64)

        cxt = tc.Context()
        cxt.x1 = tc.tensor.Dense.load(x.shape, tc.I64, x.flatten().tolist())
        cxt.x2 = cxt.x1.split(3, axis=0)
        cxt.result = [cxt.x2[i].shape for i in range(3)]
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, [[shape[0] // splits, 30]] * splits)

    def testLogarithm(self):
        size = 1_000_000
        shape = [10, size / 10]

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.arange(shape, 2, size + 2)
        cxt.ln = cxt.x.log()
        cxt.log = cxt.x.log(math.e)
        cxt.test = (cxt.ln == cxt.log).all()

        self.assertTrue(self.host.post(ENDPOINT, cxt))

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

    def testMean(self):
        shape = [2, 3, 4]
        axis = 1
        x = (np.random.random(np.product(shape)) * 10).reshape(shape)

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(shape, tc.F32, x.flatten().tolist())
        cxt.result = cxt.x.mean(axis)

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.mean(x, axis)
        self.assertTrue(allClose(actual, expected))

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
        self.assertEqual(actual, np.product(range(1, 7)))

    def testRound(self):
        shape = [10, 20]
        x = (np.random.random(np.product(shape)) * 10).reshape(shape)

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(shape, tc.F32, x.flatten().tolist())
        cxt.result = cxt.x.round()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(tc.I32, shape, x.round().astype(np.int).flatten()))

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

    def testSliceAndTransposeAndSliceAndSlice(self):
        self.maxDiff = None
        shape = [2, 3, 4, 5]

        cxt = tc.Context()
        cxt.big = tc.tensor.Dense.arange(shape, 0, 120)
        cxt.medium = cxt.big[0]
        cxt.small = cxt.medium.transpose()[1, 1:3]
        cxt.tiny = cxt.small[0, :-1]

        expected = np.arange(0, 120).reshape(shape)
        expected = expected[0]
        expected = np.transpose(expected)[1, 1:3]
        expected = expected[0, :-1]

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(tc.I64, expected.shape, expected.flatten()))

    def testFlip(self):
        shape = [5, 4, 3]

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.arange(shape, 0, 60)
        cxt.result = cxt.x.flip(0)

        expected = np.arange(0, 60).reshape(shape)
        expected = np.flip(expected, 0)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(tc.I64, expected.shape, expected.flatten()))

    def testReshape(self):
        source = [2, 3, 4, 1]
        dest = [3, 8]

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.arange(source, 0, 24)
        cxt.result = cxt.x.reshape(dest)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(tc.I64, dest, np.arange(24).tolist()))

    def testTile(self):
        shape = [2, 3]
        multiples = 2

        x = np.arange(0, np.product(shape)).reshape(shape)

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.I32)
        cxt.result = tc.tensor.tile(cxt.x, 2)

        actual = self.host.post(ENDPOINT, cxt)

        expected = np.tile(x, multiples)
        self.assertEqual(actual, expect_dense(tc.I32, list(expected.shape), expected.flatten().tolist()))

    def testArgmax(self):
        shape = [2, 3, 4]

        x = np.arange(0, np.product(shape)).reshape(shape)

        cxt = tc.Context()
        cxt.x = load_dense(x)
        cxt.am = cxt.x.argmax()
        cxt.am0 = cxt.x.argmax(0)
        cxt.am1 = cxt.x.argmax(1)
        cxt.result = cxt.am, cxt.am0, cxt.am1

        actual_am, actual_am0, actual_am1 = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual_am, np.argmax(x))
        self.assertEqual(actual_am0, expect_dense(tc.U64, [3, 4], np.argmax(x, 0).flatten().tolist()))
        self.assertEqual(actual_am1, expect_dense(tc.U64, [2, 4], np.argmax(x, 1).flatten().tolist()))

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
        cxt.result = tc.After(cxt.tensor[coord].write(value), cxt.tensor)

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_sparse(tc.I32, shape, [[coord, value]])
        self.assertEqual(actual, expected)

    def testWriteAndSlice(self):
        shape = [2, 5]

        cxt = tc.Context()
        cxt.tensor = tc.tensor.Sparse.zeros(shape)
        cxt.result = tc.After(cxt.tensor[:, 2:-1].write(1), cxt.tensor)

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_sparse(tc.F32, shape, [[[0, 2], 1], [[0, 3], 1], [[1, 2], 1], [[1, 3], 1]])
        self.assertEqual(actual, expected)

    def testAdd(self):
        shape = [5, 2, 3]

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape)
        cxt.small = tc.tensor.Sparse.zeros([3])
        cxt.result = tc.After([
            cxt.big[1].write(1),
            cxt.small[1].write(2),
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
            cxt.big[:2].write(1),
            cxt.small[0].write(-2),
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
            cxt.big[:, 1:-2].write(2),
            cxt.small[1].write(3),
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
            cxt.big[:, 1:-2].write(2),
            cxt.small[1].write(3),
        ], cxt.small - cxt.big)

        actual = self.host.post(ENDPOINT, cxt)

        big = np.zeros(shape)
        big[:, 1:-2] = 2
        small = np.zeros([5, 2])
        small[1] = 3
        expected = small - big

        expected = expect_sparse(tc.I32, shape, expected)
        self.assertEqual(actual, expected)

    def testSum(self):
        shape = [2, 4, 3, 5]
        axis = 1

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape, tc.I32)
        cxt.result = tc.After(cxt.big[0, 1:3].write(2), cxt.big.sum(axis))

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.zeros(shape, dtype=np.int32)
        expected[0, 1:3] = 2
        expected = expected.sum(axis)
        expected = expect_sparse(tc.I32, [2, 3, 5], expected)
        self.assertEqual(actual, expected)

    def testProduct(self):
        shape = [2, 4, 3, 5]
        axis = 2

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape, tc.I32)
        cxt.result = tc.After(cxt.big[0, 1:3].write(2), cxt.big.product(axis))

        actual = self.host.post(ENDPOINT, cxt)

        expected = np.zeros(shape, dtype=np.int32)
        expected[0, 1:3] = 2
        expected = expected.prod(axis)
        expected = expect_sparse(tc.I32, [2, 4, 5], expected)

        self.assertEqual(actual, expected)

    def testSliceAndBroadcast(self):
        self.maxDiff = None
        data = [
            [[0, 0, 3, 0], 1.],
            [[0, 2, 0, 0], 2.],
            [[1, 0, 0, 0], 3.],
        ]
        shape = [2, 5, 2, 3, 4, 10]

        cxt = tc.Context()
        cxt.small = tc.tensor.Sparse.load([2, 3, 4, 1], tc.I32, data)
        cxt.big = cxt.small * tc.tensor.Dense.ones(shape, tc.I32)
        cxt.slice = cxt.big[:-1, 1:4, 1]

        actual = self.host.post(ENDPOINT, cxt)

        expected = np.zeros([2, 3, 4, 1])
        for coord, value in data:
            expected[tuple(coord)] = value
        expected = expected * np.ones(shape)
        expected = expected[:-1, 1:4, 1]

        expected = expect_sparse(tc.I32, expected.shape, expected)
        self.assertEqual(actual, expected)

    def testArgmax(self):
        shape = [2, 3]

        x = (np.random.random(np.product(shape)) * 2) - 1
        x = (x * (np.abs(x) > 0.5)).reshape(shape)

        cxt = tc.Context()
        cxt.x = tc.tensor.Sparse.load(
            shape, tc.F32, [(list(coord), x[coord]) for coord in np.ndindex(x.shape) if x[coord] != 0])

        cxt.am = cxt.x.argmax()
        cxt.am0 = cxt.x.argmax(0)
        cxt.result = cxt.am, cxt.am0

        actual_am, actual_am0 = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual_am, np.argmax(x))
        self.assertEqual(actual_am0, expect_sparse(tc.U64, [3], np.argmax(x, 0)))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_tensor", cache_size="1G")

    def testAdd(self):
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([3, 5, 2], 0, 30)
        cxt.sparse = tc.tensor.Sparse.zeros([5, 2], tc.I32)
        cxt.result = tc.After(cxt.sparse[1].write(3), cxt.dense + cxt.sparse)

        actual = self.host.post(ENDPOINT, cxt)
        l = np.arange(0, 30).reshape([3, 5, 2])
        r = np.zeros([5, 2], np.int32)
        r[1] = 3
        expected = l + r
        self.assertEqual(actual, expect_dense(tc.I64, [3, 5, 2], expected.flatten()))

    def testDiv(self):
        self.maxDiff = None
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([30, 3, 2], 1., 181.)
        cxt.sparse = tc.tensor.Sparse.zeros([3, 2], tc.F32)
        cxt.result = tc.After(cxt.sparse[1].write(2), cxt.sparse / cxt.dense)

        actual = self.host.post(ENDPOINT, cxt)
        l = np.arange(1, 181).reshape([30, 3, 2])
        r = np.zeros([3, 2], np.float)
        r[1] = 2.
        expected = r / l
        self.assertEqual(actual, expect_sparse(tc.F64, [30, 3, 2], expected))

    def testMul(self):
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([3], 0, 3)
        cxt.sparse = tc.tensor.Sparse.zeros([2, 3], tc.I32)
        cxt.result = tc.After(cxt.sparse[0, 1:3].write(2), cxt.dense * cxt.sparse)

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.zeros([2, 3])
        expected[0, 1:3] = 2
        expected = expected * np.arange(0, 3)
        self.assertEqual(actual, expect_sparse(tc.I64, [2, 3], expected))

    def testSubAndSum(self):
        x = 300
        y = 250
        z = 2

        cxt = tc.Context()
        cxt.sparse = tc.tensor.Sparse.zeros([1, y, z])
        cxt.dense = tc.tensor.Dense.ones([x, 1, z], tc.I32)
        cxt.result = (cxt.sparse - cxt.dense).sum()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, -(x * y * z))

    def testSparseAsDense(self):
        matrix = np.eye(3).astype(np.bool)
        data = [(list(coord), bool(matrix[coord])) for coord in np.ndindex(matrix.shape) if matrix[coord] != 0]

        cxt = tc.Context()
        cxt.sparse = tc.tensor.Sparse.load([3, 3], tc.Bool, data)
        cxt.dense = cxt.sparse.as_dense()

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.Bool, [3, 3], matrix.flatten().tolist())
        self.assertEqual(actual, expected)

    def testDenseAsSparse(self):
        matrix = np.eye(3).astype(np.int)

        cxt = tc.Context()
        cxt.dense = load_dense(matrix, tc.I32)
        cxt.sparse = cxt.dense.as_sparse()

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_sparse(tc.I32, [3, 3], matrix)
        self.assertEqual(actual, expected)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


class ChainTests(PersistenceTest, unittest.TestCase):
    CACHE_SIZE = "100M"
    NUM_HOSTS = 4
    NAME = "tensor"

    def cluster(self, chain_type):
        class Persistent(tc.Cluster, metaclass=tc.Meta):
            __uri__ = tc.URI(f"http://127.0.0.1:{DEFAULT_PORT}/test/tensor")

            def _configure(self):
                schema = tc.tensor.Schema([2, 3], tc.I32)
                self.dense = chain_type(tc.tensor.Dense(schema))
                self.sparse = chain_type(tc.tensor.Sparse(schema))

            @tc.put_method
            def overwrite(self, txn):
                txn.new = tc.tensor.Dense.constant([3], 2)
                return [
                    self.dense.write(txn.new),
                    self.sparse[0].write(txn.new)
                ]

            @tc.get_method
            def eq(self):
                return self.sparse == self.dense

        return Persistent

    def execute(self, hosts):
        hosts[0].put("/test/tensor/dense", [0, 0], 1)
        hosts[1].put("/test/tensor/sparse", [0, 0], 1)

        dense = expect_dense(tc.I32, [2, 3], [1, 0, 0, 0, 0, 0])
        sparse = expect_sparse(tc.I32, [2, 3], [[[0, 0], 1]])
        for host in hosts:
            actual = host.get("/test/tensor/dense")
            self.assertEqual(actual, dense)

            actual = host.get("/test/tensor/sparse")
            self.assertEqual(actual, sparse)

        hosts[1].stop()
        hosts[0].put("/test/tensor/overwrite")
        hosts[1].start()

        dense = expect_dense(tc.I32, [2, 3], [2] * 6)

        expected = np.zeros([2, 3])
        expected[0] = (np.ones([3]) * 2)
        sparse = expect_sparse(tc.I32, [2, 3], expected)

        eq = expect_dense(tc.Bool, [2, 3], [True, True, True, False, False, False])

        for host in hosts:
            actual = host.get("/test/tensor/dense")
            self.assertEqual(actual, dense)

            actual = host.get("/test/tensor/sparse")
            self.assertEqual(actual, sparse)

            actual = host.get("/test/tensor/eq")
            self.assertEqual(actual, eq)


def allClose(actual, expected):
    return np.allclose(actual[tc.uri(tc.tensor.Dense)][1], expected.flatten())


def expect_dense(dtype, shape, flat):
    return {
        str(tc.uri(tc.tensor.Dense)): [
            [list(shape), str(tc.uri(dtype))],
            list(flat),
        ]
    }


def expect_sparse(dtype, shape, values):
    if isinstance(values, np.ndarray):
        values = nparray_to_sparse(values, dtype)

    return {
        str(tc.uri(tc.tensor.Sparse)): [
            [list(shape), str(tc.uri(dtype))],
            list(values),
        ]
    }


def load_dense(x, dtype=tc.F32):
    return tc.tensor.Dense.load(x.shape, dtype, x.flatten().tolist())


def nparray_to_sparse(arr, dtype):
    dtype = float if issubclass(dtype, tc.Float) else int
    zero = dtype(0)
    coords = itertools.product(*[range(dim) for dim in arr.shape])
    sparse = [[list(coord), n] for (coord, n) in zip(coords, (dtype(n) for n in arr.flatten())) if n != zero]
    return sparse


if __name__ == "__main__":
    unittest.main()
