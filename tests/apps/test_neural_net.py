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
PADDING = 1
STRIDE = 1


class TestLibrary(tc.app.Library):
    __uri__ = URI

    @staticmethod
    def uses():
        return {
            "ML": tc.hosted_ml.service.ML
        }

    @tc.post_method
    def check_conv_layer(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        input_shape, output_shape = CNN_SHAPE[0]
        cxt.layer = self.ML.ConvLayer.create(input_shape, output_shape, PADDING, STRIDE)
        return cxt.layer.forward(inputs=inputs)

    @tc.post_method
    def check_dnn_layer(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.layer = self.ML.DNNLayer.create(2, 2, self.ML.Sigmoid())
        return cxt.layer.forward(inputs=inputs)

    @tc.post_method
    def check_cnn(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.nn = self.create_cnn()
        return cxt.nn.forward(inputs=inputs)

    @tc.get_method
    def create_cnn(self) -> tc.hosted_ml.nn.NeuralNet:
        layers = tc.Tuple([self.ML.ConvLayer.create(i, o, PADDING, STRIDE) for i, o in CNN_SHAPE])
        return self.ML.Sequential(layers)

    @tc.post_method
    def check_dnn(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.nn = self.create_dnn()
        return cxt.nn.forward(inputs=inputs)

    @tc.get_method
    def create_dnn(self) -> tc.hosted_ml.nn.NeuralNet:
        layers = tc.Tuple([self.ML.DNNLayer.create(i, o, self.ML.Sigmoid()) for i, o in DNN_SHAPE])
        return self.ML.Sequential(layers)


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_docker("test_neural_net", [tc.hosted_ml.service.ML(), TestLibrary()])

    def testConvLayer(self):
        input_shape = [BATCH_SIZE] + CNN_SHAPE[0][0]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        output = self.host.post(URI.append("check_conv_layer"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 2, 3, 3])

    def testCNN(self):
        input_shape = [BATCH_SIZE] + CNN_SHAPE[0][0]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        tc.print_json(self.host.get(URI.append("create_cnn")))
        output = self.host.post(URI.append("check_cnn"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 2, 1, 1])

    def testDNN(self):
        input_shape = [BATCH_SIZE, DNN_SHAPE[0][0]]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        output = self.host.post(URI.append("check_dnn"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 2])

    def testDNNLayer(self):
        input_shape = [BATCH_SIZE, DNN_SHAPE[0][0]]
        inputs = np.random.random(np.product(input_shape)).reshape(input_shape)
        output = self.host.post(URI.append("check_dnn_layer"), {"inputs": load_dense(inputs)})
        self.assertEqual(output_shape(output), [BATCH_SIZE, 1])


def load_dense(nparray):
    return tc.tensor.Dense.load(nparray.shape, tc.F32, nparray.flatten().tolist())


def output_shape(as_json):
    return as_json[tc.uri(tc.tensor.Dense)][0][0]


if __name__ == "__main__":
    unittest.main()
