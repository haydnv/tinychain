import tinychain as tc
import unittest

from testutils import start_host


ENDPOINT = "/transact/hypothetical"


@tc.get_op
def example(txn) -> tc.Number:
    txn.a = tc.Number(5) # this is a State
    txn.b = tc.Number(10) # this is a State
    txn.product = txn.a * txn.b # this is a Ref
    return txn.product


@tc.get_op
def to_feet(txn, meters: tc.Number) -> tc.Number:
    # IMPORTANT! don't use Python's if statement! use tc.If!
    return tc.If(
        meters >= 0,
        meters * 3.28,
        tc.error.BadRequest("negative distance is not supported"))


class ClientDocTests(unittest.TestCase):
    def setUp(self):
        self.host = start_host("test_client_docs")

    def testHello(self):
        hello = "Hello, World!"
        self.assertEqual(self.host.post(ENDPOINT, tc.String(hello)), hello)

    def testExampleOp(self):
        cxt = tc.Context()
        cxt.example = example
        cxt.result = cxt.example()
        self.assertEqual(self.host.post(ENDPOINT, cxt), 50)

    def tearDown(self):
        self.host.stop()


if __name__ == "__main__":
    unittest.main()

