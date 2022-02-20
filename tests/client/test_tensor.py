import numpy as np
import tinychain as tc
import unittest

from testutils import ClientTest

ENDPOINT = "/transact/hypothetical"
SIZE = 10
EXPECTED = np.array([i if i % 2 == 0 else i * 10 for i in range(1, SIZE + 1)])


# in this case, we know the `size` of the Tensor at compile time,
# so it's safe to use a regular Python for loop, and we don't need an Op decorator like `@tc.post_op`
def example1(tensor, size):
    # Tensor.write is a PUT Op that we're using for its side-effects
    writes = [tensor[i].write(tensor[i] * 10) for i in range(size) if i % 2 == 0]
    # so we have to use `After` explicitly to make sure the writes are executed before the `tensor` is returned
    return tc.After(writes, tensor)


# in this case, we don't know the size of the Tensor at compile time,
# so we have to call `Stream.range` at runtime to achieve the same result
#
# since this function `example2` is only called at compile time though,
# it still doesn't need a decorator like `@tc.post_op`
def example2(tensor):
    # the `@tc.closure` decorator captures referenced states from the outer context
    # in this case "tensor"
    @tc.closure(tensor)
    @tc.get_op  # since the `step` Op will be called at runtime, it needs a decorator
    def step(i: tc.U64):
        return tensor[i].write(tensor[i] * 10)

    # this creates a new `Stream` with a range of numbers, and executes `step` for each `i` in the `Stream`
    return tc.Stream.range((0, tensor.size, 2)).for_each(step)


# `example3` needs its own Op context at runtime, so we define it as a POST Op using the `@tc.post_op` decorator
@tc.post_op
def example3(x: tc.tensor.Dense):  # without this type annotation, TinyChain won't know what type of `x` to expect
    @tc.closure(x)
    @tc.post_op
    def cond(i: tc.UInt):
        return i < x.size

    @tc.closure(x)
    @tc.post_op
    def step(i: tc.UInt):
        write = x[i].write(x[i] * 10)
        return tc.After(write, tc.Map(i=i + 2))  # return the new state of the loop

    return tc.While(cond, step, tc.Map(i=0))


class TensorTests(ClientTest):
    @staticmethod
    def context():
        cxt = tc.Context()
        cxt.tensor = tc.tensor.Dense.arange([SIZE], 1, SIZE + 1)
        return cxt

    def testExample1(self):
        cxt = self.context()
        cxt.result = example1(cxt.tensor, 10)

        self.execute(cxt)

    def testExample2(self):
        cxt = self.context()
        # since we're using `example2` for its side effects, we have to use `After` explicitly
        cxt.result = tc.After(example2(cxt.tensor), cxt.tensor)

        self.execute(cxt)

    def testExample3(self):
        cxt = self.context()
        # since `example3` needs to be called at runtime, we have to explicitly include it in the test context
        cxt.loop = example3
        cxt.result = tc.After(cxt.loop(x=cxt.tensor), cxt.tensor)

        self.execute(cxt)

    def execute(self, cxt):
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(EXPECTED))


@tc.post_op
def transpose(cxt, a: tc.tensor.Dense) -> tc.tensor.Dense:
    """
    Example of a matrix transpose implemented using a nested loop.
    This is not intended to be performant. Use `Tensor.transpose` to transpose a `Tensor`.
    """

    m, n = a.shape.unpack(2)

    # this creates one `Tensor` in this `Op` context, to write to
    cxt.transposed = tc.tensor.Dense.zeros(tc.Tuple([n, m]))

    # this is a tensor creation `Op` itself, i.e. each usage of `transposed` would create a new tensor
    # transposed = tc.tensor.Dense.zeros([n, m])

    @tc.closure(a, cxt.transposed)
    @tc.get_op
    def row_step(x: tc.U64):

        @tc.closure(a, x, cxt.transposed)
        @tc.get_op
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


def expect_dense(x, dtype=tc.I64):
    return {tc.uri(tc.tensor.Dense): [[list(x.shape), tc.uri(dtype)], x.flatten().tolist()]}


if __name__ == "__main__":
    unittest.main()
