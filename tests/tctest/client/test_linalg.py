import typing as t

import numpy as np
import tinychain as tc
import unittest

from .base import ClientTest

ENDPOINT = "/transact/hypothetical"
TENSOR_URI = str(tc.uri(tc.tensor.Dense))


class LinearAlgebraTests(ClientTest):
    def testDiagonal(self):
        x = np.arange(0, 9).reshape(3, 3)

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.I32)
        cxt.diag = tc.linalg.diagonal(cxt.x)

        expected = np.diag(x)
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(expected, tc.I32))

    def testSetDiagonal(self):
        size = 3
        shape = [size, size]
        x = np.arange(0, size**2).reshape(shape)
        diag = [2] * size

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.I32)
        cxt.diag = tc.tensor.Dense.load([size], diag, tc.I32)
        cxt.result = tc.After(tc.linalg.set_diagonal(cxt.x, cxt.diag), cxt.x)

        x[range(size), range(size)] = diag
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(x, tc.I32))

    def testMatmul(self):
        l = np.random.random([2, 3, 4])
        r = np.random.random([2, 4, 5])

        cxt = tc.Context()
        cxt.l = tc.tensor.Dense.load(l.shape, l.flatten().tolist(), tc.F32)
        cxt.r = tc.tensor.Dense.load(r.shape, r.flatten().tolist(), tc.F32)
        cxt.result = cxt.l @ cxt.r

        expected = np.matmul(l, r)
        actual = self.host.post(ENDPOINT, cxt)
        actual = actual[tc.uri(tc.tensor.Dense)][1]

        self.assertTrue(np.allclose(expected.flatten(), actual))

    def testNorm(self):
        shape = [2, 3, 4]
        matrices = np.arange(24).reshape(shape)

        cxt = tc.Context()
        cxt.matrices = tc.tensor.Dense.load(shape, matrices.flatten().tolist(), tc.I32)
        cxt.result = tc.linalg.batch_norm(cxt.matrices)

        expected = [np.linalg.norm(matrix) for matrix in matrices]

        actual = self.host.post(ENDPOINT, cxt)
        actual = actual[tc.uri(tc.tensor.Dense)][1]

        self.assertEqual(actual, expected)


def expect_dense(x, dtype):
    return {tc.uri(tc.tensor.Dense): [[list(x.shape), tc.uri(dtype)], x.flatten().tolist()]}


def load_np(as_json: t.Dict[str, t.Any], dtype=np.float32) -> np.ndarray:
    shape = as_json[TENSOR_URI][0][0]
    return np.array(as_json[TENSOR_URI][1], dtype).reshape(shape)


if __name__ == "__main__":
    unittest.main()
