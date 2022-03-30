import numpy as np
import unittest
import testutils
import tinychain as tc


URI = tc.URI("/test/ml/app")
BATCH_SIZE = 20


class NeuralNetTester(tc.app.Library):
    __uri__ = URI

    @staticmethod
    def uses():
        return {
            "ML": tc.ml.service.ML
        }

    @tc.post
    def test_cnn_layer(self, cxt, inputs: tc.tensor.Tensor, labels: tc.tensor.Tensor) -> tc.F32:
        layer = tc.ml.nn.ConvLayer.create([3, 5, 5], [2, 1, 1])
        cxt.optimizer = tc.ml.optimizer.GradientDescent(layer)
        return cxt.optimizer.train(1, inputs, labels)

    @tc.post
    def test_cnn(self, cxt, inputs: tc.tensor.Tensor, labels: tc.tensor.Tensor) -> tc.F32:
        layers = [
            tc.ml.nn.ConvLayer.create([3, 5, 5], [2, 1, 1], activation=tc.ml.sigmoid),
            tc.ml.nn.ConvLayer.create([2, 3, 3], [2, 1, 1])
        ]

        cnn = tc.ml.nn.Sequential(layers)
        cxt.optimizer = tc.ml.optimizer.GradientDescent(cnn)
        return cxt.optimizer.train(1, inputs, labels)

    @tc.post
    def test_dnn_layer(self, cxt, inputs: tc.tensor.Tensor, labels: tc.tensor.Tensor) -> tc.F32:
        layer = tc.ml.nn.DNNLayer.create(2, 1, tc.ml.sigmoid)
        cxt.optimizer = tc.ml.optimizer.GradientDescent(layer)
        return cxt.optimizer.train(1, inputs, labels)

    @tc.post
    def test_dnn(self, cxt, inputs: tc.tensor.Tensor, labels: tc.tensor.Tensor) -> tc.F32:
        layers = [
            tc.ml.nn.DNNLayer.create(2, 3, tc.ml.sigmoid),
            tc.ml.nn.DNNLayer.create(3, 5),
            tc.ml.nn.DNNLayer.create(5, 1, tc.ml.sigmoid)]

        dnn = tc.ml.nn.Sequential(layers)
        cxt.optimizer = tc.ml.optimizer.Adam(dnn)
        return cxt.optimizer.train(1, inputs, labels)


class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_docker("test_neural_net", NeuralNetTester())

    def testCNN(self):
        inputs = np.ones([BATCH_SIZE, 3, 5, 5])

        self.host.post(tc.uri(NeuralNetTester).append("test_cnn_layer"), {
            "inputs": load_dense(inputs),
            "labels": load_dense(np.ones([BATCH_SIZE, 2, 3, 3]) * 2),
        })

        self.host.post(tc.uri(NeuralNetTester).append("test_cnn"), {
            "inputs": load_dense(inputs),
            "labels": load_dense(np.ones([BATCH_SIZE, 2, 2, 2]) * 2),
        })

    def testDNN(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        labels = np.logical_or(inputs[:, 0], inputs[:, 1]).reshape([BATCH_SIZE, 1])

        self.host.post(tc.uri(NeuralNetTester).append("test_dnn_layer"), {
            "inputs": load_dense(inputs),
            "labels": load_dense(labels),
        })

        self.host.post(tc.uri(NeuralNetTester).append("test_dnn"), {
            "inputs": load_dense(inputs),
            "labels": load_dense(labels),
        })

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.uri(tc.tensor.Dense)][0][0]


if __name__ == "__main__":
    unittest.main()
