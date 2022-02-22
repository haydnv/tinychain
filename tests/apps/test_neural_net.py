import unittest
import testutils
import tinychain as tc


URI = tc.URI("/test/ml/app")
LAYER_SHAPE = [[2, 2], [2, 1]]


class TestLibrary(tc.app.Library):
    __uri__ = URI

    @staticmethod
    def uses():
        return {
            "ML": tc.hosted_ml.service.ML
        }

    @tc.post_method
    def check_layer(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.dnn_layer = self.ML.DNNLayer.create(2, 2, self.ML.Sigmoid())
        return cxt.dnn_layer.forward(inputs=inputs)

    @tc.post_method
    def check_dnn(self, cxt, inputs: tc.tensor.Tensor) -> tc.tensor.Tensor:
        cxt.dnn = self.create_dnn()
        return cxt.dnn.forward(inputs=inputs)

    @tc.get_method
    def create_dnn(self) -> tc.hosted_ml.nn.NeuralNet:
        layers = tc.Tuple([self.ML.DNNLayer.create(i, o, self.ML.Sigmoid()) for i, o in LAYER_SHAPE])
        return self.ML.Sequential(layers)


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_docker("test_neural_net", [tc.hosted_ml.service.ML(), TestLibrary()])

    def testDNNLayer(self):
        print(self.host.post(URI.append("check_layer"), {"inputs": tc.tensor.Dense.load([1, 2], tc.F32, [0, 1])}))

    def testDNN(self):
        tc.print_json(self.host.get(URI.append("create_dnn")))
        print(self.host.post(URI.append("check_dnn"), {"inputs": tc.tensor.Dense.load([1, 2], tc.F32, [0, 1])}))


if __name__ == "__main__":
    unittest.main()
