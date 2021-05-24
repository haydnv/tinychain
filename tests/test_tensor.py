import tinychain as tc
import unittest

from testutils import PORT, start_host, PersistenceTest


ENDPOINT = "/transact/hypothetical"


class DenseTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_dense_tensor")

    def testConstant(self):
        const = 1.414

        cxt = tc.Context()
        cxt.tensor = tc.Tensor.Dense.constant((3, 2, 1), 1.414)

        expected = {
            '/state/collection/tensor/dense': [
                ['/state/scalar/value/number/float/64', [3, 2, 1]],
                [const] * 6,
            ]
        }
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expected, actual)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()

