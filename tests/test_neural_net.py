import numpy as np
import tinychain as tc
import unittest

from testutils import start_host


Dense = tc.tensor.Dense


ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = 0.01


class Activation(tc.Object):
    @tc.post_method
    def forward(self, Z: Dense) -> Dense:
        raise NotImplemented

    @tc.post_method
    def backward(self, dA: Dense, Z: Dense) -> Dense:
        raise NotImplemented


class Sigmoid(Activation):
    @tc.post_method
    def forward(self, Z: Dense) -> Dense:
        return 1 / (1 + (-Z).exp())

    @tc.post_method
    def backward(self, dA: Dense, Z: Dense) -> Dense:
        sig = self.forward(Z=Z)
        return dA * sig * (1 - sig)


class ReLU(Activation):
    @tc.post_method
    def forward(self, Z: Dense) -> Dense:
        return Z * (Z > 0)

    @tc.post_method
    def backward(self, dA: Dense, Z: Dense) -> Dense:
        return dA * (Z > 0)


class Layer(tc.Object):
    @tc.attribute
    def activation(self) -> Activation:
        pass

    @tc.attribute
    def bias(self) -> Dense:
        pass

    @tc.attribute
    def weights(self) -> Dense:
        pass

    @tc.post_method
    def eval(self, cxt, inputs: Dense) -> Dense:
        cxt.dot = inputs * self.weights.transpose()
        cxt.Z = (cxt.dot + self.bias).sum(1)
        return self.activation.forward(Z=cxt.Z)


def layer_init(input_size, output_size, activation):
    shape = (input_size, output_size)
    bias = np.random.random([output_size])
    weights = np.random.random(shape)
    return {
        "bias": Dense.load([output_size], tc.F32, bias.tolist()),
        "weights": Dense.load(shape, tc.F32, weights.flatten().tolist()),
        "activation": activation,
    }


class NeuralNet(tc.Tuple):
    def eval(self, inputs):
        @tc.post_op
        def eval_layer(item: Layer, state: Dense):
            return item.eval(inputs=state)

        return self.fold(inputs.expand_dims(-1), eval_layer)

    def error(self, inputs, labels):
        output = self.eval(inputs)
        return (output - labels)**2


class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_nn", overwrite=True)

    def testXOR(self):
        cxt = tc.Context()
        cxt.Sigmoid = Sigmoid
        cxt.ReLU = ReLU
        cxt.Layer = Layer

        cxt.inputs = Dense.load([4, 2], tc.Bool, [
            False, False,
            False, True,
            True, False,
            True, True,
        ])
        cxt.labels = Dense.load([4], tc.Bool, [False, True, True, False])

        cxt.l1 = layer_init(2, 2, cxt.ReLU)
        cxt.l2 = layer_init(2, 1, cxt.Sigmoid)
        cxt.nn = NeuralNet([tc.New(cxt.Layer, cxt.l1), tc.New(cxt.Layer, cxt.l2)])

        cxt.result = cxt.nn.error(cxt.inputs, cxt.labels)

        response = self.host.post(ENDPOINT, cxt)
        contents = response[str(tc.uri(Dense))]
        self.assertEqual(contents[0], [[4], str(tc.uri(tc.F64))])
        self.assertEqual(len(contents[1]), 4)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
