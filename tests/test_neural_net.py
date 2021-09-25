import numpy as np
import tinychain as tc
import unittest

from testutils import start_host


ENDPOINT = "/transact/hypothetical"


@tc.post_op
def sigmoid(Z: tc.tensor.Dense):
    return 1 / (1 + (-Z).exp())


@tc.post_op
def relu(Z: tc.tensor.Dense):
    return Z * (Z > 0)


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
        cxt.result = cxt.nn.eval(cxt.inputs)

        print(tc.to_json(cxt))
        response = self.host.post(ENDPOINT, cxt)
        contents = response[str(tc.uri(tc.tensor.Dense))]
        self.assertEqual(contents[0], [[4], str(tc.uri(tc.F64))])
        self.assertEqual(len(contents[1]), 4)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
