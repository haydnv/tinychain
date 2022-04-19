import itertools
import random

import numpy as np
import tinychain as tc
import unittest

from .base import HostTest

ENDPOINT = "/transact/hypothetical"


class EinsumTests(HostTest):
    def testExpandDims(self):
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([2, 3], 0, 6)
        cxt.result = cxt.dense.expand_dims(1)

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(np.arange(0, 6).reshape([2, 1, 3]))
        self.assertEqual(actual, expected)

    def testTranspose(self):
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([3, 2], 0, 6)
        cxt.result = cxt.dense.transpose()

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.transpose(np.arange(0, 6).reshape([3, 2]))
        expected = expect_dense(expected)
        self.assertEqual(actual, expected)

    def test1D(self):
        A = np.array([1, 2, 3])
        self.execute('i->i', A)

    def test2Dx1(self):
        A = np.array([[1, 1], [2, 2], [3, 3]])
        self.execute('ij->i', A)
        self.execute('ij->j', A)
        self.execute('ij->ij', A)
        self.execute('ij->ji', A)

    def test2Dto3D(self):
        A = np.array([[0, 1], [1, 2], [2, 3]])
        self.execute('ij,ik->ijk', A, A)

    def test2Dx2(self):
        def _execute(A, B):
            self.execute("ij,jk->ijk", A, B)
            self.execute("ij,jk->ij", A, B)
            self.execute("ij,jk->ji", A, B)
            self.execute("ij,jk->ik", A, B)
            self.execute("ij,jk->ki", A, B)
            self.execute("ij,jk->jk", A, B)
            self.execute("ij,jk->kj", A, B)
            self.execute("ij,jk->i", A, B)
            self.execute("ij,jk->j", A, B)
            self.execute("ij,jk->k", A, B)

        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        _execute(A, B)

        A = np.array([[2], [1]])
        B = np.array([[7, 0]])
        _execute(A, B)

        A = np.array([[1, 0]])
        B = np.array([[0], [3]])
        _execute(A, B)

        A = np.array([[1, 0], [0, 1]])
        B = np.array([[4, 7], [0, 2]])
        _execute(A, B)

        A = np.array([[1, 1, 0], [0, 3, 1]])
        B = np.array([[2, 5], [1, 2], [2, 5]])
        _execute(A, B)

    def test3Dx2(self):
        A = np.array([[1, 1, 1],
                      [2, 2, 2],
                      [5, 5, 5]])

        B = np.array([[0, 1, 0],
                      [1, 1, 0],
                      [1, 1, 1]])

        self.execute('ij,jk->ijk', A, B)
        self.execute('ij,jk->ik', A, B)

        A = np.arange(60).reshape(3, 4, 5)
        B = np.arange(24).reshape(4, 3, 2)
        self.execute('ijk,jil->il', A, B)
        self.execute('ijk,jil->kj', A, B)
        self.execute('ijk,jil->lkij', A, B)
        self.execute('ijk,jil->lij', A, B)

    def test2DRandom(self):
        rand_dim = lambda x: random.randint(1, x)

        for i in range(1, 4):
            x = rand_dim(i)
            y = rand_dim(i)
            A = (np.random.random([x, y]) * 4).astype(np.int32)

            self.execute("ij->i", A)
            self.execute("ij->j", A)
            self.execute("ij->ij", A)
            self.execute("ij->ji", A)

            B = (np.random.random([y, x]) * 8).astype(np.int32)

            self.execute("ij,jk->ijk", A, B)
            self.execute("ij,jk->ik", A, B)
            self.execute("ij,jk->ki", A, B)
            self.execute("ij,jk->i", A, B)
            self.execute("ij,jk->j", A, B)
            self.execute("ij,jk->k", A, B)

    def execute(self, fmt, *tensors):
        expected = np.einsum(fmt, *[np.array(t) for t in tensors])

        cxt = tc.Context()
        cxt.dense = [to_dense(t) for t in tensors]
        cxt.sparse = [to_sparse(t) for t in tensors]
        cxt.results = (tc.tensor.einsum(fmt, cxt.dense), tc.tensor.einsum(fmt, cxt.sparse))

        (dense, sparse) = self.host.post(ENDPOINT, cxt)

        # print("inputs:")
        # for tensor in tensors:
        #     print(tensor.shape)
        #     print(tensor)
        #     print()

        # print("expect", expected.shape, expected)
        # print()
        # print("expect dense", expect_dense(expected))
        # print("actual dense", dense)
        # print()
        # print("expect sparse", expect_sparse(expected))
        # print("actual sparse", sparse)

        if expected.shape:
            self.assertEqual(dense, expect_dense(expected))
            self.assertEqual(sparse, expect_sparse(expected))
        else:
            self.assertEqual(dense, expected)
            self.assertEqual(sparse, expected)


def expect_dense(ndarray):
    shape = list(ndarray.shape)
    dtype = np_to_tc_dtype(ndarray.dtype)

    return {
        str(tc.uri(tc.tensor.Dense)): [
            [shape, str(tc.uri(dtype))],
            ndarray.flatten().tolist(),
        ]
    }


def expect_sparse(ndarray):
    shape = list(ndarray.shape)
    dtype = np_to_tc_dtype(ndarray.dtype)

    coords = itertools.product(*[range(dim) for dim in shape])
    elements = [
        [list(coord), n]
        for (coord, n) in zip(coords, (n for n in ndarray.flatten().tolist())) if n != 0]

    return {
        str(tc.uri(tc.tensor.Sparse)): [
            [shape, str(tc.uri(dtype))],
            elements,
        ]
    }


def to_dense(ndarray):
    dtype = np_to_tc_dtype(ndarray.dtype)
    shape = list(ndarray.shape)
    return tc.tensor.Dense.load(shape, [int(n) for n in ndarray.flatten()], dtype)


def to_sparse(ndarray):
    dtype = np_to_tc_dtype(ndarray.dtype)
    shape = list(ndarray.shape)

    data = []
    for coord in itertools.product(*[range(x) for x in ndarray.shape]):
        value = ndarray[coord]
        if value:
            data.append([coord, int(value)])

    return tc.tensor.Sparse.load(shape, data, dtype)


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
