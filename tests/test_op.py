import tinychain as tc
import unittest

from base import InstanceTest


@tc.Op.Get
def cast_i64(cxt, n):
    return tc.OpRef.Get(tc.I64.PATH, n)


class ValueTests(InstanceTest):
#    def testCastI64(self):
#        expected = tc.to_json(cast_i64)
#        actual = self.host.resolve(cast_i64)
#        self.assertEqual(expected, actual)

    def testCall(self):
        cxt = tc.Context()
        cxt.test_op = cast_i64
        cxt.result = cxt.test_op(3.14)
 
        actual = self.host.resolve(cxt)
        self.assertEqual(actual, 3)


if __name__ == "__main__":
    unittest.main()

