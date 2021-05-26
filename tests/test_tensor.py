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

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


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

