import tinychain as tc
import unittest

from testutils import start_host


ENDPOINT = "/transact/hypothetical"


@tc.get_op
def meters_to_feet(txn, value: tc.Number):
    return value * 3.28


class OpTests(unittest.TestCase):
    def setUp(self):
        self.host = start_host("test_ops")

    def testGet(self):
        cxt = tc.Context()
        cxt.op = meters_to_feet
        cxt.result = cxt.op(2)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, 6.56)

    def tearDown(self):
        self.host.stop()


if __name__ == "__main__":
    unittest.main()

