import numpy as np
import tinychain as tc
import unittest

from testutils import PORT, start_host, PersistenceTest


ENDPOINT = "/transact/hypothetical"


class DenseTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_dense_tensor")

    def testConstant(self):
        c = 1.414
        shape = [3, 2, 1]

        cxt = tc.Context()
        cxt.tensor = tc.Tensor.Dense.constant(shape, c)
        cxt.result = tc.After(cxt.tensor.write([0, 0, 0], 0), cxt.tensor)

        expected = expect(tc.F64, shape, [0] + [c] * (product(shape) - 1))
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expected, actual)

    def testSlice(self):
        shape = [2, 5]

        cxt = tc.Context()
        cxt.tensor = tc.Tensor.Dense.arange(shape, 1, 11)
        cxt.result = cxt.tensor[1, 2:-1]

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect(tc.I64, [2], np.arange(1, 11).reshape([2, 5])[1, 2:-1])
        self.assertEqual(actual, expected)

    def testAdd(self):
        shape = [5, 2, 1]

        cxt = tc.Context()
        cxt.left = tc.Tensor.Dense.arange(shape, 1., 6.)
        cxt.right = tc.Tensor.Dense.constant([5], 2)
        cxt.result = cxt.left + cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect(tc.F64, shape, np.arange(1, 6, 0.5) + 2)
        self.assertEqual(actual, expected)

    def testDiv(self):
        shape = [3]

        cxt = tc.Context()
        cxt.left = tc.Tensor.Dense.arange(shape, 2., 8.)
        cxt.right = tc.Tensor.Dense.constant([1], 2)
        cxt.result = cxt.left / cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect(tc.F64, shape, np.arange(1, 4))
        self.assertEqual(actual, expected)

    def testMul(self):
        shape = [5, 2, 1]

        cxt = tc.Context()
        cxt.left = tc.Tensor.Dense.arange(shape, 1, 11)
        cxt.right = tc.Tensor.Dense.constant([5], 2)
        cxt.result = cxt.left * cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect(tc.I64, shape, np.arange(1, 11) * 2)
        self.assertEqual(actual, expected)

    def testSub(self):
        shape = [1, 3]

        cxt = tc.Context()
        cxt.left = tc.Tensor.Dense.arange(shape, 0, 6)
        cxt.right = tc.Tensor.Dense.constant([1], 2)
        cxt.result = cxt.left - cxt.right

        actual = self.host.post(ENDPOINT, cxt)
        expected = expect(tc.I64, shape, np.arange(-2, 4, 2))
        self.assertEqual(actual, expected)

    def testLogic(self):
        big = [20, 20, 10]
        trailing = [10]

        cxt = tc.Context()
        cxt.big_ones = tc.Tensor.Dense.ones(big, tc.U8)
        cxt.big_zeros = tc.Tensor.Dense.zeros(big, tc.U8)
        cxt.true = tc.Tensor.Dense.ones(trailing)
        cxt.false = tc.Tensor.Dense.zeros(trailing)
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
        shape = [2, 3]

        cxt = tc.Context()
        cxt.big = tc.Tensor.Dense.arange(shape, 1, 7)
        cxt.result = cxt.big.product()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, product(range(1, 7)))

    def testSum(self):
        shape = [5, 2]

        cxt = tc.Context()
        cxt.big = tc.Tensor.Dense.arange(shape, 0, 10)
        cxt.result = cxt.big.sum()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, sum(range(10)))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()



class ChainTests(PersistenceTest, unittest.TestCase):
    NUM_HOSTS = 4
    NAME = "tensor"

    def cluster(self, chain_type):
        class Persistent(tc.Cluster):
            __uri__ = tc.URI(f"http://127.0.0.1:{PORT}/test/tensor")

            def _configure(self):
                schema = tc.Tensor.Schema([2, 3], tc.I32)
                self.dense = chain_type(tc.Tensor.Dense(schema))

            @tc.put_method
            def overwrite(self, txn):
                txn.new = tc.Tensor.Dense.constant([3], 2)
                return self.dense.write(None, txn.new)

        return Persistent

    def execute(self, hosts):
        hosts[0].put("/test/tensor/dense", [0, 0], 1)

        for host in hosts:
            actual = host.get("/test/tensor/dense")
            expected = expect(tc.I32, [2, 3], [1, 0, 0, 0, 0, 0])
            self.assertEqual(actual, expected)

        hosts[0].put("/test/tensor/overwrite")
        expected = expect(tc.I32, [2, 3], [2] * 6)
        actual = hosts[0].get("/test/tensor/dense")
        self.assertEqual(actual, expected)


def expect(dtype, shape, flat):
    return {
        str(tc.uri(tc.Tensor.Dense)): [
            [shape, str(tc.uri(dtype))],
            list(flat),
        ]
    }


def product(seq):
    p = 1
    for n in seq:
        p *= n

    return p

if __name__ == "__main__":
    unittest.main()

