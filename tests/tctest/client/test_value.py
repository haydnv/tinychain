import tinychain as tc
import unittest

from templates import ClientTest

ENDPOINT = "/transact/hypothetical"


class NumberTests(ClientTest):
    def testDivideByZero(self):
        cxt = tc.Context()
        cxt.result = tc.F32(3.14) / tc.F32(0.)

        self.assertRaises(tc.error.BadRequest, lambda: self.host.post(ENDPOINT, cxt))


if __name__ == "__main__":
    unittest.main()
