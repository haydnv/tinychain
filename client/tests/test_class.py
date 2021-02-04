import tinychain as tc
import unittest


@tc.class_def
class Meters(tc.Number):
    @tc.get_method
    def feet(self) -> tc.Number:
        return self * 3.28


class ClassTests(unittest.TestCase):
    def testJson(self):
        expected = {
            '/state/scalar/value/number': {
                'feet': {'/state/scalar/op/get': ['key', [
                    ['_return', {'$self/mul': [3.28]}]]
                ]}
            }
        }

        actual = tc.to_json(Meters)
        self.assertEqual(expected, actual)

    def testInContext(self):
        expected = [
            ['M', {'/state/object/class': [
                {'/state/scalar/value/number': {
                    'feet': {'/state/scalar/op/get': ['key', [
                        ['_return', {'$self/mul': [3.28]}]]]}
                    }
                }
            ]}],
            ['m', {'$M': [2]}],
            ['ft', {'$m/feet': [None]}]
        ]

        cxt = tc.Context()
        cxt.M = tc.Class.init(Meters)
        cxt.m = Meters(tc.OpRef.Get(cxt.M, 2))
        cxt.ft = cxt.m.feet()

        actual = tc.to_json(cxt)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

