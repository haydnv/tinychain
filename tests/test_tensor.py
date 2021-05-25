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

        expected = {
            str(tc.uri(tc.Tensor.Dense)): [
                [str(tc.uri(tc.F64)), shape],
                [0.] + [c] * (product(shape) - 1)
            ]
        }

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expected, actual)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


def product(seq):
    p = 1
    for n in seq:
        p *= n

    return p

if __name__ == "__main__":
    unittest.main()

