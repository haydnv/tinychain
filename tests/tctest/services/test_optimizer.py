import numpy as np
import unittest
import time

import rjwt
import tinychain as tc

from ..process import start_host

BATCH_SIZE = 20
NS = tc.URI("/test_neural_net")


class OptimizerTester(tc.service.Library):
    NAME = "lib"
    VERSION = tc.ml.VERSION

    __uri__ = tc.service.library_uri(None, NS, NAME, VERSION)

    nn = tc.ml.NeuralNets()
    optimizers = tc.ml.Optimizers()

    @tc.post
    def test_cnn_layer(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        labels = tc.tensor.Dense.constant([2, 3, 3], 2)
        layer = self.nn.ConvLayer.create([3, 5, 5], [2, 1, 1])
        cxt.optimizer = self.optimizers.GradientDescent(layer, lambda _, o: (o - labels)**2)
        return cxt.optimizer.train(1, inputs)

    @tc.post
    def test_cnn(self, cxt, inputs: tc.tensor.Tensor) -> tc.F32:
        layers = [
            tc.ml.nn.ConvLayer.create([3, 5, 5], [2, 1, 1], activation=tc.ml.sigmoid),
            tc.ml.nn.ConvLayer.create([2, 3, 3], [5, 2, 2], activation=tc.ml.sigmoid),
        ]

        cnn = tc.ml.nn.Sequential(layers)
        outputs = cnn.eval(inputs)
        cxt.optimizer = tc.ml.optimizer.Adam(cnn, lambda _, o: (o - 2)**2)
        return cxt.optimizer.train(1, inputs)

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

        layers = [
            tc.ml.nn.Linear.create(2, 3, tc.ml.sigmoid),
            tc.ml.nn.Linear.create(3, 5),
            tc.ml.nn.Linear.create(5, 1, tc.ml.sigmoid)]

        dnn = tc.ml.nn.Sequential(layers)
        cxt.optimizer = tc.ml.optimizer.Adam(dnn, cost)
        return cxt.optimizer.train(1, inputs)


class OptimizerTests(unittest.TestCase):
    URI = tc.URI(OptimizerTester)

    @classmethod
    def setUpClass(cls):
        cls.actor = rjwt.Actor('/')
        cls.host = start_host(NS, public_key=cls.actor.public_key)

        cls.host.put(tc.URI(tc.service.Library), str(tc.ml.NS)[1:], tc.URI(tc.service.Library) + tc.ml.NS)
        cls.host.install(tc.ml.NeuralNets())
        cls.host.install(tc.ml.Optimizers())

        cls.host.put(tc.URI(tc.service.Library), cls.URI[-3], cls.URI[:-2])
        cls.host.install(OptimizerTester())

    def testCNN(self):
        inputs = np.ones([BATCH_SIZE, 3, 5, 5])
        self.host.post(self.URI.append("test_cnn_layer"), {"inputs": load_dense(inputs)})
        self.host.post(self.URI.append("test_cnn"), {"inputs": load_dense(inputs)})

    def testDNN(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])

        self.host.post(self.URI.append("test_linear"), {"inputs": load_dense(inputs)})

        start = time.time()
        self.host.post(self.URI.append("test_dnn"), {"inputs": load_dense(inputs)})
        elapsed = time.time() - start
        print(f"trained a deep neural net in {elapsed:.2}s")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.URI(tc.tensor.Dense)][0][0]
