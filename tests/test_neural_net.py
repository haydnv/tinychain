import numpy as np
import tinychain as tc
import unittest

from testutils import start_host


Dense = tc.tensor.Dense


ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = 0.01


def create_layer(input_size, output_size):
    shape = (input_size, output_size)
    # bias = np.random.random([output_size]).tolist()
    weights = tc.tensor.Dense.load(shape, tc.F32, np.random.random(input_size * output_size).tolist())
    return tc.ml.layer(weights)


class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_nn", overwrite=True)

    def testXOR(self):
        cxt = tc.Context()

        cxt.inputs = Dense.load([4, 2], tc.Bool, [
            False, False,
            False, True,
            True, False,
            True, True,
        ])
        cxt.labels = Dense.load([4], tc.Bool, [False, True, True, False])

        cxt.nn = tc.ml.neural_net([create_layer(2, 2), create_layer(2, 1)])

        cxt.result = cxt.nn.eval(cxt.inputs)
        # import json
        # print(json.dumps(tc.to_json(cxt)))

        response = self.host.post(ENDPOINT, cxt)
        # print(response)

        contents = response[str(tc.uri(Dense))]
        self.assertEqual(contents[0], [[4, 1], str(tc.uri(tc.F32))])
        self.assertEqual(len(contents[1]), 4)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
