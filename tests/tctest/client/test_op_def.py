import unittest
import tinychain as tc

HOST = tc.host.Host('http://127.0.0.1:8702')
ENDPOINT = '/transact/hypothetical'


@tc.differentiable
def f(x: tc.Numeric) -> tc.Numeric:
    return 2 * x


class OpDefTests(unittest.TestCase):
    def testDerivative(self):
        cxt = tc.Context()
        cxt.f = f
        cxt.x = tc.ml.Variable.ones([2, 2])
        cxt.f_x = cxt.f(cxt.x)
        cxt.expected = tc.math.derivative_of(cxt.f_x)
        cxt.actual = tc.math.derivative_of(cxt.f)(cxt.x)
        cxt.result = cxt.expected == cxt.actual

        self.assertTrue(HOST.post(ENDPOINT, cxt))
