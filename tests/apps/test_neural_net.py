import numpy as np
import unittest
import testutils
import tinychain as tc


URI = tc.URI("/test/ml/app")
BATCH_SIZE = 20


class TestLibrary(tc.app.Library):
    __uri__ = URI

    @staticmethod
    def uses():
        return {
            "ML": tc.hosted_ml.service.ML
        }

    @tc.post
    def test_dnn_layer(self, cxt, inputs: tc.tensor.Tensor, labels: tc.tensor.Tensor) -> tc.F32:
        layer = tc.hosted_ml.nn.DNNLayer.create(2, 1)
        cxt.optimizer = tc.hosted_ml.optimizer.GradientDescent(layer)
        return cxt.optimizer.train(inputs, labels).sum()

    @tc.post
    def test_dnn(self, cxt, inputs: tc.tensor.Tensor, labels: tc.tensor.Tensor) -> tc.F32:
        layers = [tc.hosted_ml.nn.DNNLayer.create(2, 2), tc.hosted_ml.nn.DNNLayer.create(2, 1)]
        dnn = tc.hosted_ml.nn.Sequential(layers)
        cxt.optimizer = tc.hosted_ml.optimizer.GradientDescent(dnn)
        return cxt.optimizer.train(inputs, labels).sum()

    # @tc.post
    # def test_cnn_layer(self, cxt, inputs: tc.tensor.Tensor, labels: tc.tensor.Tensor) -> tc.F32:
    #     padding, stride = 1, 1
    #     layer = [tc.hosted_ml.nn.ConvLayer.create(inputs.shape[1:], (1, 3, 3), padding=padding, stride=stride)]
    #     cnn = tc.hosted_ml.nn.Sequential(layer)
    #     cxt.optimizer = tc.hosted_ml.optimizer.GradientDescent(cnn)
    #     print(layer)
    #     return cxt.optimizer.train(inputs, labels).sum()


class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_docker("test_neural_net", [tc.hosted_ml.service.ML(), TestLibrary()])

    def testDNNLayer(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        labels = np.logical_or(inputs[:, 0], inputs[:, 1])

        self.host.post(tc.uri(TestLibrary).append("test_dnn_layer"), {
            "inputs": load_dense(inputs),
            "labels": load_dense(labels),
        })

    def testDNN(self):
        inputs = np.random.random(2 * BATCH_SIZE).reshape([BATCH_SIZE, 2])
        labels = np.logical_or(inputs[:, 0], inputs[:, 1])

        self.host.post(tc.uri(TestLibrary).append("test_dnn"), {
            "inputs": load_dense(inputs),
            "labels": load_dense(labels),
        })

    # def testConvLayer(self):
    #     inputs = np.random.random(3*5*5*BATCH_SIZE).reshape((BATCH_SIZE, 3, 5, 5))
    #     labels = np.expand_dims(inputs.sum(axis=1), 1)

    #     self.host.post(tc.uri(TestLibrary).append("test_test_cnn_layer"), {
    #         "inputs": load_dense(inputs),
    #         "labels": load_dense(labels),
    #     })

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.uri(tc.tensor.Dense)][0][0]


if __name__ == "__main__":
    unittest.main()
