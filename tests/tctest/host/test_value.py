import tinychain as tc
import unittest

from .base import HostTest

ENDPOINT = "/transact/hypothetical"


class ValueTests(HostTest):
    def testComplexNumber(self):
        cxt = tc.Context()
        cxt.c = tc.C32((-1.23, 3.14))

        actual = self.host.post(ENDPOINT, cxt)
        expected = complex(-1.23, 3.14)
        self.assertEqual(parse(actual), expected)


def parse(as_json):
    dtype = next(iter(as_json.keys()))
    assert dtype.startswith(str(tc.URI(tc.Complex)))
    return complex(*as_json[dtype])


if __name__ == "__main__":
    unittest.main()
