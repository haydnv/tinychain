import tinychain as tc
import unittest

from base import InstanceTest


@tc.Op.Get
def cast_i64(cxt, n):
    return tc.OpRef.Get(tc.I64.PATH, n)


class ValueTests(InstanceTest):
    def testCastI64(self):
        expected = tc.to_json(cast_i64)
        actual = self.host.resolve(cast_i64)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

