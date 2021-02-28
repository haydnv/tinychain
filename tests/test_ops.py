import tinychain as tc
import unittest

from testutils import start_host


ENDPOINT = "/transact/hypothetical"


@tc.get_op
def meters_to_feet(txn, value: tc.Number) -> tc.Number:
    return tc.If(
        value >= 0,
        value * 3.28,
        tc.error.BadRequest("negative distance is not supported"))


@tc.post_op
def echo(txn, foo) -> tc.Map:
    return tc.Map({"foo": foo})


class OpTests(unittest.TestCase):
    def setUp(self):
        self.host = start_host("test_ops")

    def testGet(self):
        cxt = tc.Context()
        cxt.op = meters_to_feet
        cxt.result = cxt.op(2)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, 6.56)

        cxt = tc.Context()
        cxt.op = meters_to_feet
        cxt.result = cxt.op(-2)

        with self.assertRaises(tc.error.BadRequest):
            self.host.post(ENDPOINT, cxt)

    def testPost(self):
        cxt = tc.Context()
        cxt.op = echo
        cxt.result = cxt.op(foo="bar")

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, {"foo": "bar"})

    def tearDown(self):
        self.host.stop()


if __name__ == "__main__":
    unittest.main()

