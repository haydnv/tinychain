import typing as t

import numpy as np
import tinychain as tc
import unittest

from .base import ClientTest

TENSOR_URI = str(tc.URI(tc.tensor.Dense))


class LinearAlgebraTests(ClientTest):
    def testDiagonal(self):
        x = np.arange(0, 9).reshape(3, 3)

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.I32)
        cxt.diag = tc.math.linalg.diagonal(cxt.x)

        expected = np.diag(x)
        actual = self.host.hypothetical(cxt)
        self.assertEqual(actual, expect_dense(expected, tc.I32))

    def testWithDiagonal(self):
        size = 3
        shape = [size, size]
        x = np.arange(0, size**2).reshape(shape)
        diag = [2] * size

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.I32)
        cxt.diag = tc.tensor.Dense.load([size], diag, tc.I32)
        cxt.result = tc.math.linalg.with_diagonal(cxt.x, cxt.diag)

        x[range(size), range(size)] = diag
        actual = self.host.hypothetical(cxt)
        self.assertEqual(actual, expect_dense(x, tc.I32))


def expect_dense(x, dtype):
    return {
        str(tc.URI(tc.tensor.Dense)): [
            [str(tc.URI(dtype)), list(x.shape)],
            x.flatten().tolist(),
        ]
    }


def load_np(as_json: t.Dict[str, t.Any], dtype=np.float32) -> np.ndarray:
    shape = as_json[TENSOR_URI][0][0]
    return np.array(as_json[TENSOR_URI][1], dtype).reshape(shape)


if __name__ == "__main__":
    unittest.main()
