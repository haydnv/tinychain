import tinychain as tc
import unittest

from base import InstanceTest


values = [
    (tc.String, "Hello, world!"),
    (tc.Number, -1.414),
    (tc.U8, 0),
    (tc.U16, 1),
    (tc.U32, 2),
    (tc.U64, 3),
    (tc.UInt, 4),
    (tc.I16, -2),
    (tc.I32, -1),
    (tc.I64, 0),
    (tc.Int, 1),
    (tc.F32, -0.23),
    (tc.F64, 1.23),
    (tc.Float, 3.14),
    (tc.C32, [1.414, -2]),
    (tc.C64, [0, 1]),
    (tc.Link, "http://www.example.com/path"),
]


class ValueTests(InstanceTest):
    def testEach(self):
        for dtype, spec in values:
            expect = tc.to_json(dtype(spec))
            self.assertEqual(self.host.get(dtype.PATH, spec), expect)

    def testAll(self):
        actual = self.host.resolve([dtype(spec) for dtype, spec in values])
        expected = tc.to_json([dtype(spec) for dtype, spec in values])

        for actual, expected in zip(actual, expected):
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()

