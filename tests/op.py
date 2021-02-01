import tinychain as tc
import unittest

from base import InstanceTest


@tc.Op.Get
def cast_i64(cxt, n):
    return tc.OpRef.Get(tc.I64.PATH, n)


class ValueTests(InstanceTest):
    def testCastI64(self):
        print(tc.to_json(cast_i64))
        print(self.host.resolve(cast_i64))


if __name__ == "__main__":
    unittest.main()

