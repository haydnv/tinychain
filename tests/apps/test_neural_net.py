import numpy as np
import unittest
import testutils
import tinychain as tc


URI = tc.URI("/test/ml/app")
BATCH_SIZE = 1

DNN_SHAPE = [[2, 2], [2, 1]]

CNN_SHAPE = [
    [[3, 5, 5], [2, 1, 1]],
    [[2, 3, 3], [1, 3, 3]],
]


class TestLibrary(tc.app.Library):
    __uri__ = URI

    @staticmethod
    def uses():
        return {
            "ML": tc.hosted_ml.service.ML
        }

    @tc.get
    def create_dnn_layer(self) -> tc.hosted_ml.nn.Layer:
        return tc.hosted_ml.nn.DNNLayer.create(2, 2, tc.ml.Sigmoid())

    @tc.get
    def load_dnn_layer(self, cxt) -> tc.hosted_ml.nn.Layer:
        cxt.weights = tc.tensor.Dense.create(DNN_SHAPE[0])
        cxt.bias = tc.tensor.Dense.create(DNN_SHAPE[0][1:]) + 1
        return tc.hosted_ml.nn.DNNLayer(cxt.weights, cxt.bias)

    @tc.post
    def check_dnn_layer(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.layer = self.create_dnn_layer()
        return cxt.layer.forward(inputs=inputs)

    @tc.post
    def check_conv_layer(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        input_shape, output_shape = CNN_SHAPE[0]
        cxt.layer = tc.hosted_ml.nn.ConvLayer.create(input_shape, output_shape)
        return cxt.layer.forward(inputs=inputs)

    @tc.post
    def check_conv_layer_zero_padded(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        input_shape = [20, 1, 2]
        output_shape = [4, 1, 1]
        cxt.layer = tc.hosted_ml.nn.ConvLayer.create(input_shape, output_shape, padding=0)
        return cxt.layer.forward(inputs=inputs)

    @tc.post
    def check_cnn(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.nn = self.create_cnn()
        return cxt.nn.forward(inputs=inputs)

    @tc.get
    def create_cnn(self) -> tc.hosted_ml.nn.NeuralNet:
        layers = tc.Tuple([tc.hosted_ml.nn.ConvLayer.create(i, o) for i, o in CNN_SHAPE])
        return tc.hosted_ml.nn.Sequential(layers)

    @tc.get
    def create_dnn(self) -> tc.hosted_ml.nn.NeuralNet:
        return tc.hosted_ml.nn.DNN.create([(i, o, tc.ml.Sigmoid()) for i, o in DNN_SHAPE])

    @tc.post
    def check_dnn(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.nn = self.create_dnn()
        return cxt.nn.forward(inputs=inputs)


class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_docker("test_neural_net", [tc.hosted_ml.service.ML(), TestLibrary()])

    def testConvLayer(self):
        input_shape = [BATCH_SIZE] + CNN_SHAPE[0][0]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        output = self.host.post(URI.append("check_conv_layer"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 2, 3, 3])

    def testConvLayerWithZeroPadding(self):
        input_shape = [BATCH_SIZE, 20, 1, 2]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        output = self.host.post(URI.append("check_conv_layer_zero_padded"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 4, 1, 1])

    def testCNN(self):
        # tc.print_json(self.host.get(URI.append("create_cnn")))

        input_shape = [BATCH_SIZE] + CNN_SHAPE[0][0]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        output = self.host.post(URI.append("check_cnn"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 1, 1, 1])

    def testDNN(self):
        # tc.print_json(self.host.get(URI.append("create_dnn")))

        input_shape = [BATCH_SIZE, DNN_SHAPE[0][0]]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        output = self.host.post(URI.append("check_dnn"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 1])

    def testDNNLayer(self):
        # tc.print_json(self.host.get(URI.append("create_dnn_layer")))

        input_shape = [BATCH_SIZE, DNN_SHAPE[0][0]]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        output = self.host.post(URI.append("check_dnn_layer"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 2])

    def testLoadFromOuterContext(self):
        layer = self.host.get(URI.append("load_dnn_layer"))
        bias = layer[tc.uri(tc.Instance)][tc.uri(tc.hosted_ml.nn.Layer)]["bias"]
        self.assertTrue(tc.uri(tc.tensor.Dense) in bias)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.uri(tc.tensor.Dense)][0][0]


if __name__ == "__main__":
    unittest.main()
