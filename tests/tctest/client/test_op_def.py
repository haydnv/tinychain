import unittest
import tinychain as tc

HOST = tc.host.Host('http://127.0.0.1:8702')
ENDPOINT = '/transact/hypothetical'


@tc.differentiable
def f(x: tc.tensor.Tensor) -> tc.tensor.Tensor:
    return 3 * x**2


class OpDefTests(unittest.TestCase):
    def testDerivative(self):
        cxt = tc.Context()
        cxt.x = tc.ml.Variable.ones([2, 2])
        cxt.f = f
        cxt.d_f = tc.math.derivative_of(cxt.f)
        cxt.f_x = cxt.f(cxt.x)
        cxt.actual = cxt.d_f(cxt.x)
        cxt.expected = tc.math.derivative_of(cxt.f_x)
        cxt.result = (cxt.expected == (6 * cxt.x)).logical_and(cxt.expected == cxt.actual)

        self.assertTrue(HOST.post(ENDPOINT, cxt))
