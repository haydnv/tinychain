import math
import numpy as np
import tinychain as tc
import unittest

from testutils import start_host


ENDPOINT = "/transact/hypothetical"


@tc.post_op
def sigmoid(cxt, Z: tc.tensor.Dense):
    e = tc.tensor.Dense.constant(Z.shape(), math.e)
    one = tc.tensor.Dense.constant(Z.shape(), 1)
    zero = tc.tensor.Dense.constant(Z.shape(), 1)
    cxt.pow = zero - Z
    return one / (one + e**cxt.pow)


@tc.post_op
def relu(Z: tc.tensor.Dense):
    zero = tc.tensor.Dense.constant(Z.shape(), 0)
    return Z * (Z > zero)


def nn_layer(input_size, output_size, activation):
    shape = (input_size, output_size)
    bias = np.random.random([output_size])
    weights = np.random.random(shape)
    return tc.Map({
        "bias": tc.tensor.Dense.load([output_size], tc.F32, bias.tolist()),
        "weights": tc.tensor.Dense.load(shape, tc.F32, weights.flatten().tolist()),
        "activation": activation,
    })


class NeuralNet(tc.Tuple):
    def eval(self, inputs):
        @tc.post_op
        def eval_layer(cxt, item: tc.Map, state: tc.tensor.Dense):
            weights = tc.tensor.Dense(item["weights"])

            cxt.activation = tc.op.Post(item["activation"])
            cxt.weights = weights.transpose()
            cxt.dot = (state * cxt.weights).sum(1)
            cxt.Z = cxt.dot + item["bias"]
            return cxt.activation(Z=cxt.Z)

        return self.fold(inputs, eval_layer)


class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_nn", overwrite=True)

    def testXOR(self):
        cxt = tc.Context()
        cxt.sigmoid = sigmoid
        cxt.relu = relu
        cxt.inputs = tc.tensor.Dense.load([1, 8], tc.U8, [
            0, 1, 0, 1,
            0, 0, 1, 1
        ])
        cxt.nn = NeuralNet([nn_layer(8, 8, cxt.relu), nn_layer(8, 4, cxt.sigmoid)])
        # cxt.inputs = tc.tensor.Dense.load([1, 1], tc.U8, [0])
        # cxt.nn = NeuralNet([nn_layer(1, 1, cxt.relu)])
        cxt.result = cxt.nn.eval(cxt.inputs)
        print(tc.to_json(cxt))
        print(self.host.post(ENDPOINT, cxt))

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
