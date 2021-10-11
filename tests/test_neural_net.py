import numpy as np
import tinychain as tc
import unittest

from testutils import start_host


Dense = tc.tensor.Dense


ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = 0.01


def create_layer(input_size, output_size, activation):
    shape = (input_size, output_size)
    bias = tc.tensor.Dense.load([output_size], tc.F32, np.random.random([output_size]).tolist())
    weights = tc.tensor.Dense.load(shape, tc.F32, np.random.random(input_size * output_size).tolist())
    return tc.ml.layer(weights, bias, activation)


class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_nn", overwrite=True, cache_size="1G")

    def testXOR(self):
        cxt = tc.Context()

        cxt.inputs = Dense.load([4, 2], tc.Bool, [
            False, False,
            False, True,
            True, False,
            True, True,
        ])

        cxt.labels = Dense.load([4], tc.Bool, [False, True, True, False])

        cxt.layer1 = create_layer(2, 2, tc.ml.ReLU())
        cxt.layer2 = create_layer(2, 1, tc.ml.Sigmoid())
        cxt.nn = tc.ml.neural_net([cxt.layer1, cxt.layer2], LEARNING_RATE)

        @tc.closure
        @tc.post_op
        def train(i: tc.UInt):
            return tc.After(cxt.nn.train(cxt.inputs, cxt.labels), {"i": i + 1})

        cxt.training = tc.While(tc.closure(tc.post_op(lambda i: tc.UInt(i) < 25)), train, {"i": 0})

        error = (cxt.nn.eval(cxt.inputs) - cxt.labels.expand_dims())**2
        cxt.result = tc.After(cxt.training, error)

        response = self.host.post(ENDPOINT, cxt)

        contents = response[str(tc.uri(Dense))]
        self.assertEqual(contents[0], [[4, 1], str(tc.uri(tc.F64))])
        self.assertEqual(len(contents[1]), 4)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
