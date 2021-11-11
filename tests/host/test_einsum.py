import itertools
import random

import numpy as np
import tinychain as tc
import unittest

from testutils import start_host


ENDPOINT = "/transact/hypothetical"


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_einsum", cache_size="1G")

    def testExpandDims(self):
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([2, 3], 0, 6)
        cxt.result = cxt.dense.expand_dims(1)

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.I64, [2, 1, 3], range(6))
        self.assertEqual(actual, expected)

    def testTranspose(self):
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([3, 2], 0, 6)
        cxt.result = cxt.dense.transpose()

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.transpose(np.arange(0, 6).reshape([3, 2]))
        expected = expect_dense(tc.I64, [2, 3], expected.flatten())
        self.assertEqual(actual, expected)

    def test1D(self):
        A = np.array([1, 2, 3])
        self.execute('i->', A)
        self.execute('i->i', A)

    def testRepeatIndex(self):
        A = np.array([[1, 1], [2, 2]])
        B = np.array([3, 2, 1])
        self.execute('ii,j->ij', A, B)

    def test2D(self):
        A = np.array([[1, 1], [2, 2], [3, 3]])
        self.execute('ij->', A)
        self.execute('ij->i', A)
        self.execute('ij->j', A)
        self.execute('ij->ij', A)
        self.execute('ij->ji', A)

    def test2Dto3D(self):
        A = np.array([[0, 1], [1, 2], [2, 3]])
        self.execute('ij,ik->ijk', A, A)
        self.execute('ij,ik->', A, A)

    def test2DMulti(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        self.execute('ik,ik,il->', A, A, A)
        self.execute('ji,jk->ik', A, B)  # = matmul(transpose(A), B)
        self.execute('ij,jk->ijk', A, B)
        self.execute('ij,jk->ij', A, B)
        self.execute('ij,jk->ik', A, B)
        self.execute('ij,jk->jk', A, B)
        self.execute('ij,jk->i', A, B)
        self.execute('ij,jk->j', A, B)
        self.execute('ij,jk->k', A, B)
        self.execute('ij,jk->', A, B)

        A = np.array([[1], [2], [3], [4]])
        B = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.execute("kj,ki->ij", A, B)

    def test3DMulti(self):
        A = np.array([[1, 1, 1],
                      [2, 2, 2],
                      [5, 5, 5]])

        B = np.array([[0, 1, 0],
                      [1, 1, 0],
                      [1, 1, 1]])

        self.execute('ij,jk->ijk', A, B)
        self.execute('ij,jk->ik', A, B)

        A = np.arange(60.).reshape(3, 4, 5)
        B = np.arange(24.).reshape(4, 3, 2)
        self.execute('ijk,jil->', A, B)
        self.execute('ijk,jil->il', A, B)
        self.execute('ijk,jil->kj', A, B)
        self.execute('ijk,jil->lkij', A, B)
        self.execute('ijk,jil->lij', A, B)

    def test2DRandom(self):
        rand_dim = lambda x: random.randint(1, x)

        for i in range(1, 10):
            x = rand_dim(i)
            y = rand_dim(i)
            A = (np.random.random([x, y]) * 256).astype(np.int32)

            self.execute("ij->i", A)

            B = (np.random.random([y, x]) * 512).astype(np.int32)

            self.execute("ij,jk->ik", A, B)
            self.execute("ij,jk->ki", A, B)

    def execute(self, fmt, *tensors):
        expected = np.einsum(fmt, *[np.array(t) for t in tensors])

        cxt = tc.Context()
        cxt.dense = [to_dense(t) for t in tensors]
        cxt.sparse = [to_sparse(t) for t in tensors]
        cxt.results = (tc.tensor.einsum(fmt, cxt.dense), tc.tensor.einsum(fmt, cxt.sparse))

        (dense, sparse) = self.host.post(ENDPOINT, cxt)
        self.assertEqual(dense, to_dense(expected))
        self.assertEqual(sparse, to_sparse(expected))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


def expect_dense(dtype, shape, flat):
    return {
        str(tc.uri(tc.tensor.Dense)): [
            [shape, str(tc.uri(dtype))],
            list(flat),
        ]
    }


def to_dense(ndarray):
    dtype = np_to_tc_dtype(ndarray.dtype)
    shape = list(ndarray.shape)
    return tc.tensor.Dense.load(shape, dtype, [int(n) for n in ndarray.flatten()])


def to_sparse(ndarray):
    dtype = np_to_tc_dtype(ndarray.dtype)
    shape = list(ndarray.shape)

    data = []
    for coord in itertools.product(*[range(x) for x in ndarray.shape]):
        value = ndarray[coord]
        if value:
            data.append([coord, int(value)])

    return tc.tensor.Sparse.load(shape, dtype, data)


def np_to_tc_dtype(dtype):
    if dtype == np.float64:
        return tc.F64
    elif dtype == np.int32:
        return tc.I32
    elif dtype == np.int64:
        return tc.I64
    else:
        raise NotImplementedError(f"numpy dtype conversion of {dtype}")


if __name__ == "__main__":
    unittest.main()
