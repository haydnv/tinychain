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
    def test_convolution(self, inputs: tc.tensor.Tensor) -> tc.F32:
        layer = self.ml.ConvLayer.create([3, 5, 5], [2, 1, 1])
        return layer.eval(inputs)

    @tc.post
    def test_linear(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        cxt.layer = tc.ml.nn.Linear.create(2, 1)
        return cxt.layer.eval(inputs)

    @tc.post
    def test_derivative(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        cxt.layer = tc.ml.nn.Linear.create(2, 1)
        cxt.outputs = cxt.layer.eval(inputs)
        return tc.math.gradients(cxt.outputs, tc.tensor.Dense.ones_like(cxt.outputs), [cxt.layer.weights, cxt.layer.bias])

    @tc.post
    def test_sequential(self, inputs: tc.tensor.Tensor) -> tc.F32:
        layer1 = tc.ml.nn.Linear.create(2, 2)
        layer2 = tc.ml.nn.Linear.create(2, 1)
        return tc.ml.nn.Sequential([layer1, layer2]).eval(inputs)

    @tc.post
    def test_sequential_derivative(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        cxt.layer1 = tc.ml.nn.Linear.create(2, 3)
        cxt.layer2 = tc.ml.nn.Linear.create(3, 2)
        cxt.layer3 = tc.ml.nn.Linear.create(2, 1)
        cxt.outputs = tc.ml.nn.Sequential([cxt.layer1, cxt.layer2, cxt.layer3]).eval(inputs)
        vars = [cxt.layer1.weights, cxt.layer1.bias, cxt.layer2.weights, cxt.layer2.bias, cxt.layer3.weights, cxt.layer3.bias]
        grads = tc.math.gradients(cxt.outputs, tc.tensor.Dense.ones_like(cxt.outputs), vars)
        cxt.grads = grads
        assert len(vars) == len(cxt.grads)
        return [grad.shape for grad in cxt.grads]
        # return [var.update(grad.sum(0)) for (var, grad) in zip(vars, grads)]


class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_neural_net", NeuralNetTester(), wait_time=2, request_ttl=60)

    def testConvolution(self):
        inputs = np.ones([BATCH_SIZE, 3, 5, 5])
        self.host.post(URI.append("test_convolution"), {"inputs": load_dense(inputs)})

    def testLinear(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        self.host.post(URI.append("test_linear"), {"inputs": load_dense(inputs)})

    def testDerivative(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        self.host.post(URI.append("test_derivative"), {"inputs": load_dense(inputs)})

    def testSequential(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        self.host.post(URI.append("test_sequential"), {"inputs": load_dense(inputs)})

    def testSequentialDerivative(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        grads = self.host.post(URI.append("test_sequential_derivative"), {"inputs": load_dense(inputs)})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.URI(tc.tensor.Dense)][0][0]
