import numpy as np
import tinychain as tc
import unittest

from testutils import start_host


Dense = tc.tensor.Dense


ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = 0.01
MAX_ITERATIONS = 500


def create_layer(input_size, output_size, activation):
    shape = (input_size, output_size)
    bias = tc.tensor.Dense.load([output_size], tc.F32, np.random.random([output_size]).tolist())
    weights = tc.tensor.Dense.load(shape, tc.F32, np.random.random(input_size * output_size).tolist())
    return tc.ml.layer(weights, bias, activation)


# TODO: optimize Tensor reduce ops & implement AdamOptimizer
# so these tests will finish within the request timeout
@unittest.skip
class NeuralNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host("test_nn", overwrite=True, cache_size="1G")

    def testNot(self):
        cxt = tc.Context()

        cxt.inputs = Dense.load([2, 1], tc.Bool, [True, False])
        cxt.labels = Dense.load([2, 1], tc.Bool, [False, True])

        cxt.input_layer = create_layer(1, 1, tc.ml.Sigmoid())
        cxt.nn = tc.ml.neural_net([cxt.input_layer])

        self.execute(cxt)

    def testIdentity(self):
        cxt = tc.Context()

        cxt.inputs = Dense.load([2, 1], tc.Bool, [True, False])
        cxt.labels = Dense.load([2, 1], tc.Bool, [True, False])

        cxt.input_layer = create_layer(1, 1, tc.ml.Sigmoid())
        cxt.nn = tc.ml.neural_net([cxt.input_layer])

        self.execute(cxt)

    def testOr(self):
        cxt = tc.Context()

        cxt.inputs = Dense.load([4, 2], tc.Bool, [
            True, True,
            True, False,
            False, True,
            False, False,
        ])
        cxt.labels = Dense.load([4, 1], tc.Bool, [True, True, True, False])

        cxt.input_layer = create_layer(2, 1, tc.ml.ReLU())
        cxt.nn = tc.ml.neural_net([cxt.input_layer])

        self.execute(cxt)

    def execute(self, cxt):
        @tc.closure
        @tc.post_op
        def while_cond(output: Dense, i: tc.UInt):
            fit = ((output > 0.5) != cxt.labels).any()
            return fit.logical_and(i < MAX_ITERATIONS)

        @tc.closure
        @tc.post_op
        def train(i: tc.UInt):
            output = cxt.nn.train(cxt.inputs, lambda output: ((output - cxt.labels)**2) * LEARNING_RATE)
            return {"output": output, "i": i + 1}

        cxt.result = tc.While(while_cond, train, {"output": cxt.nn.eval(cxt.inputs), "i": 0})

        response = self.host.post(ENDPOINT, cxt)
        self.assertLess(response["i"], MAX_ITERATIONS, "failed to converge")

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
