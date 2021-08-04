import tinychain as tc
import unittest

from testutils import start_host


ENDPOINT = "/transact/hypothetical"


class NumberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_values")

    def testDivideByZero(self):
        cxt = tc.Context()
        cxt.result = tc.F32(3.14) / tc.F32(0.)

        self.assertRaises(tc.error.BadRequest, lambda: self.host.post(ENDPOINT, cxt))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
