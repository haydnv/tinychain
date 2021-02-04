import tinychain as tc
import unittest


@tc.Class
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

    def testMethod(self):
        pass

    def testMethodFromForeignContext(self):
        pass

    def testInstanceVarInSelfContext(self):
        pass

    def testInstanceVarInForeignContext(self):
        pass


if __name__ == "__main__":
    unittest.main()

