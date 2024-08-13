import numpy as np
import tinychain as tc
import unittest

from ..process import start_host

BATCH_SIZE = 20
LIB_URI = tc.URI(tc.ml.NeuralNets)
NS = tc.URI("/test_neural_net")


class NeuralNetTester(tc.service.Library):
    NAME = "lib"
    VERSION = tc.ml.VERSION

    __uri__ = tc.service.library_uri(None, NS, NAME, VERSION)

    nn = tc.ml.service.NeuralNets()

    @tc.post
    def test_convolution(self, inputs: tc.tensor.Tensor) -> tc.F32:
        layer = self.nn.ConvLayer.create([3, 5, 5], [2, 1, 1])
        return layer.eval(inputs)

    @tc.post
    def test_linear(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        cxt.layer = tc.ml.nn.Linear.create(2, 1)
        return cxt.layer.eval(inputs)

    @tc.post
    def test_gradients(self, cxt, inputs: tc.tensor.Tensor) -> tc.State:
        cxt.layer = tc.ml.nn.Linear.create(2, 1)
        cxt.outputs = cxt.layer.eval(inputs)
        grads = cxt.layer.gradient(inputs, tc.tensor.Dense.ones_like(cxt.outputs))
        return grads["weights"], grads["bias"]

    @tc.post
    def test_sequential(self, inputs: tc.tensor.Tensor) -> tc.F32:
        layer1 = tc.ml.nn.Linear.create(2, 2)
        layer2 = tc.ml.nn.Linear.create(2, 1)
        return tc.ml.nn.Sequential([layer1, layer2]).eval(inputs)


class NeuralNetTests(unittest.TestCase):
    URI = tc.URI(NeuralNetTester)

    @classmethod
    def setUpClass(cls):
        cls.host = start_host(NS)
        cls.host.install(tc.ml.NeuralNets())
        cls.host.install(NeuralNetTester())

    def testConvolution(self):
        inputs = np.ones([BATCH_SIZE, 3, 5, 5])
        self.host.post(self.URI.append("test_convolution"), {"inputs": load_dense(inputs)})

    def testLinear(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        self.host.post(self.URI.append("test_linear"), {"inputs": load_dense(inputs)})

    def testGradients(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        response = self.host.post(self.URI.append("test_gradients"), {"inputs": load_dense(inputs)})

    def testSequential(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        self.host.post(self.URI.append("test_sequential"), {"inputs": load_dense(inputs)})

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.URI(tc.tensor.Dense)][0][0]
