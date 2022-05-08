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
        cxt.x2 = cxt.x1.split(3, axis=0)
        cxt.result = [tc.tensor.Tensor(cxt.x2[i]).shape for i in range(3)]
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, [[shape[0] // splits, 30]] * splits)

    def testSplitBySizes(self):
        input_shape = [3, 4, 1, 1]
        split = [2, 2]
        axis = 1

        x = np.random.random(np.product(input_shape)).reshape(input_shape)

        cxt = tc.Context()
        cxt.input = tc.tensor.Dense.load(x.shape, x.flatten().tolist())
        cxt.splits = cxt.input.split(split, axis=axis)
        cxt.result = [cxt.splits[i].shape for i in range(len(split))]

        actual = self.host.post(ENDPOINT, cxt)

        expected = [input_shape] * len(split)
        for i, dim in enumerate(split):
            expected[i][axis] = dim

        self.assertEqual(actual, expected)

    def testWhere(self):
        size = 5
        x = np.random.random(size).astype(bool)
        a = np.random.random(size)
        b = np.random.random(size)
        expected = np.where(x, a, b)

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.Bool)
        cxt.a = load_dense(a)
        cxt.b = load_dense(b)
        cxt.result = tc.tensor.where(cxt.x, cxt.a, cxt.b)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertTrue(all_close(actual, expected))

    def testRandomUniform(self):
        minval = -1
        maxval = 3

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.random_uniform([5, 1], minval, maxval)
        cxt.result = (cxt.x >= -1).all().logical_and((cxt.x <= 3).all()).logical_and(cxt.x.mean() > 0)

        self.assertTrue(self.host.post(ENDPOINT, cxt))

    def testTruncatedNormal(self):
        tolerance = 0.5

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.truncated_normal([10, 20])
        cxt.result = cxt.x.mean(), cxt.x.std()

        mean, std = self.host.post(ENDPOINT, cxt)
        self.assertTrue(abs(mean) < tolerance)
        self.assertTrue(abs(std - 1) < tolerance)


# Example of a matrix transpose implemented using a nested loop.
# This is not intended to be performant. Use `Tensor.transpose` to transpose any `Tensor`.
@tc.post
def transpose(cxt, a: tc.tensor.Dense) -> tc.tensor.Dense:
    m, n = a.shape.unpack(2)

    # this creates one `Tensor` in this `Op` context, to write to
    cxt.transposed = tc.tensor.Dense.zeros([n, m])

    # this is a tensor creation `Op` itself, i.e. each usage of `transposed` would create a new tensor
    # transposed = tc.tensor.Dense.zeros([n, m])

    @tc.closure(a, cxt.transposed)
    @tc.get
    def row_step(x: tc.U64):

        @tc.closure(a, x, cxt.transposed)
        @tc.get
        def step(y: tc.U64):
            return cxt.transposed[y, x].write(a[x, y])

        return tc.Stream.range((0, n)).for_each(step)

    rows = tc.Stream.range((0, m)).for_each(row_step)

    return tc.If(
        a.ndim == 2,
        tc.After(rows, cxt.transposed),
        tc.error.BadRequest("this test only supports a 2D Tensor"))


class NestedLoopTests(ClientTest):
    def testTranspose(self):
        cxt = tc.Context()
        cxt.transpose = transpose
        cxt.a = tc.tensor.Dense.arange([3, 4], 0, 12)
        cxt.a_t = cxt.transpose(a=cxt.a)
        cxt.test = (cxt.a.transpose() == cxt.a_t).all()
        self.assertTrue(self.host.post(ENDPOINT, cxt))


def all_close(actual, expected):
    return np.allclose(actual[tc.uri(tc.tensor.Dense)][1], expected.flatten())


def expect_dense(x, dtype=tc.I64):
    return {tc.uri(tc.tensor.Dense): [[list(x.shape), tc.uri(dtype)], x.flatten().tolist()]}


def load_dense(x, dtype=tc.F32):
    return tc.tensor.Dense.load(x.shape, x.flatten().tolist(), dtype)


if __name__ == "__main__":
    unittest.main()
