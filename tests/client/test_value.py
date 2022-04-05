import tinychain as tc
import unittest

from testutils import ClientTest

ENDPOINT = "/transact/hypothetical"


class NumberTests(ClientTest):
    def testDivideByZero(self):
        cxt = tc.Context()
        cxt.pi = tc.F32(3.14)
        cxt.result = cxt.pi / 0

        self.assertRaises(tc.error.BadRequest, lambda: self.host.post(ENDPOINT, cxt))


if __name__ == "__main__":
    unittest.main()
