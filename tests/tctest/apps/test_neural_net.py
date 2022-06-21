import numpy as np
import unittest
import tinychain as tc

from ..process import start_host

URI = tc.URI("/test/ml/app")
BATCH_SIZE = 20


class NeuralNetTester(tc.app.Library):
    __uri__ = URI

    ml = tc.ml.service.ML()

    @tc.post
    def test_linear(self, inputs: tc.tensor.Tensor) -> tc.F32:
        layer = tc.ml.nn.Linear.create(2, 1)
        return layer.eval(inputs)

    @tc.post
    def test_derivative(self, inputs: tc.tensor.Tensor) -> tc.F32:
        layer = tc.ml.nn.Linear.create(2, 1)
        outputs = layer.eval(inputs)
        return tc.math.gradients(outputs, tc.tensor.Dense.ones_like(outputs), [layer.weights, layer.bias])


@unittest.skip  # TODO: re-enable when differentiable methods support independent namespaces
class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_neural_net", NeuralNetTester(), wait_time=2, request_ttl=60)

    def testLinear(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        self.host.post(tc.URI(NeuralNetTester).append("test_linear"), {"inputs": load_dense(inputs)})

    def testDerivative(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        self.host.post(tc.URI(NeuralNetTester).append("test_derivative"), {"inputs": load_dense(inputs)})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.URI(tc.tensor.Dense)][0][0]
