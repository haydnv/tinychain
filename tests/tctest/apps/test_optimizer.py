import numpy as np
import unittest
import time
import tinychain as tc

from ..process import start_host

URI = tc.URI("/test/ml/optimizer")
BATCH_SIZE = 20


class OptimizerTester(tc.app.Library):
    __uri__ = URI

    ml = tc.ml.service.ML()

    @tc.post
    def test_cnn_layer(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        labels = tc.tensor.Dense.constant([2, 3, 3], 2)
        layer = self.ml.ConvLayer.create([3, 5, 5], [2, 1, 1])
        cxt.optimizer = self.ml.GradientDescent(layer, lambda _, o: (o - labels)**2)
        return cxt.optimizer.train(1, inputs, BATCH_SIZE / 2)

    @tc.post
    def test_cnn(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        layers = [
            tc.ml.nn.ConvLayer.create([3, 5, 5], [2, 1, 1], activation=tc.ml.sigmoid),
            tc.ml.nn.ConvLayer.create([2, 3, 3], [5, 2, 2], activation=tc.ml.sigmoid),
        ]

        cnn = tc.ml.nn.Sequential(layers)
        cxt.optimizer = tc.ml.optimizer.Adam(cnn, lambda _, o: (o - 2)**2)
        return cxt.optimizer.train(1, inputs, BATCH_SIZE / 10)

    @tc.post
    def test_linear(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        def cost(i, o):
            labels = tc.math.constant(i[:, 0].logical_or(i[:, 1]).expand_dims())
            return (o - labels)**2

        layer = tc.ml.nn.Linear.create(2, 1)
        cxt.optimizer = tc.ml.optimizer.GradientDescent(layer, cost)
        return cxt.optimizer.train(1, inputs)

    @tc.post
    def test_dnn(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        def cost(i, o):
            labels = tc.math.constant(i[:, 0].logical_xor(i[:, 1]).expand_dims())
            return (o - labels)**2

        schema = [
            (2, 3, tc.ml.sigmoid),
            (3, 5),
            (5, 1, tc.ml.sigmoid)]

        dnn = tc.ml.nn.DNN.create(schema)
        cxt.optimizer = tc.ml.optimizer.Adam(dnn, cost)
        return cxt.optimizer.train(1, inputs)


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_optimizer", OptimizerTester(), wait_time=2, request_ttl=60)

    def testCNN(self):
        inputs = np.ones([BATCH_SIZE, 3, 5, 5])
        self.host.post(tc.URI(OptimizerTester).append("test_cnn_layer"), {"inputs": load_dense(inputs)})
        self.host.post(tc.URI(OptimizerTester).append("test_cnn"), {"inputs": load_dense(inputs)})

    def testDNN(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])

        self.host.post(tc.URI(OptimizerTester).append("test_linear"), {"inputs": load_dense(inputs)})

        start = time.time()
        self.host.post(tc.URI(OptimizerTester).append("test_dnn"), {"inputs": load_dense(inputs)})
        elapsed = time.time() - start
        print(f"trained a deep neural net in {elapsed:.2}s")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.URI(tc.tensor.Dense)][0][0]