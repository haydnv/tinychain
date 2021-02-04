import tinychain as tc
import unittest


@tc.classdef
class Meter(tc.Number):
    @tc.get_method
    def feet(self):
        return self * 3.28


class InstanceTests(unittest.TestCase):
    def testNewInstance(self):
        expected = {
            '/state/object/class': {
                '/state/scalar/value/number': {
                    'feet': [
                        ['_result', {'$self/mul': [3.28]}]
                    ]
                }
            }
        }

        actual = tc.to_json(Meter)
        self.assertEqual(expected, actual)

    def testInstanceVariable(self):
        pass

    def testMethodFromForeignContext(self):
        expected = [
            ['m', {'/state/object/class': {
                '/state/scalar/value/number': {
                    'feet': [
                        ['_result', {'$self/mul': [3.28]}]
                    ]
                }
            }}],
            ['ft', {'$m': [2]}]
        ]

        cxt = tc.Context()
        cxt.m = Meter
        cxt.ft = tc.OpRef.Get(tc.uri(cxt.m), 2)

        actual = tc.to_json(cxt)
        self.assertEqual(expected, actual)

    def testInstanceVarInSelfContext(self):
        pass

    def testInstanceVarInForeignContext(self):
        pass


if __name__ == "__main__":
    unittest.main()

