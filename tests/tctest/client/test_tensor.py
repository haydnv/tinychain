import numpy as np
import tinychain as tc
import unittest

from .base import ClientTest

ENDPOINT = "/transact/hypothetical"
SIZE = 10


class TensorTests(ClientTest):
    def testSplitByNumber(self):
        splits = 3
        shape = (6, 30)
        x = np.ones(shape, dtype=np.int64)

        cxt = tc.Context()
        cxt.x1 = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.I64)
        cxt.x2 = tc.tensor.split(cxt.x1, 3, axis=0)
        cxt.result = [tc.tensor.Tensor(cxt.x2[i]).shape for i in range(3)]
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, [[shape[0] // splits, 30]] * splits)

    def testSplitBySizes(self):
        input_shape = [3, 4, 1, 1]
        split = [2, 2]
        axis = 1

        x = np.random.random(np.prod(input_shape)).reshape(input_shape)

        cxt = tc.Context()
        cxt.input = tc.tensor.Dense.load(x.shape, x.flatten().tolist())
        cxt.splits = tc.tensor.split(cxt.input, split, axis=axis)
        cxt.result = [cxt.splits[i].shape for i in range(len(split))]

        actual = self.host.post(ENDPOINT, cxt)

        expected = [input_shape] * len(split)
        for i, dim in enumerate(split):
            expected[i][axis] = dim

        self.assertEqual(actual, expected)

    def testTruncatedNormal(self):
        tolerance = 0.5

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.truncated_normal([10, 20])
        cxt.result = cxt.x.mean(), cxt.x.std()

        response = self.host.post(ENDPOINT, cxt)
        mean, std = response
        self.assertTrue(abs(mean) < tolerance)
        self.assertTrue(abs(std - 1) < tolerance)


def all_close(actual, expected):
    return np.allclose(actual[tc.URI(tc.tensor.Dense)][1], expected.flatten())


def load_dense(x, dtype=tc.F32):
    return tc.tensor.Dense.load(x.shape, x.flatten().tolist(), dtype)


if __name__ == "__main__":
    unittest.main()
