import numpy as np
import tinychain as tc
import unittest

from testutils import ClientTest

Dense = tc.tensor.Dense


ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = tc.F32(0.01)
MAX_ITERATIONS = 3
NUM_EXAMPLES = 20


# TODO: migrate these tests to use a hosted training backend, when available
class DNNTests(ClientTest):
    @classmethod
    def setUpClass(cls):
        np.random.seed(3)
        super().setUpClass()

    def testNot(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES).reshape([NUM_EXAMPLES, 1])
        cxt.inputs = load(inputs)
        cxt.labels = load((inputs[:, :] < 0.5).astype(np.float32).reshape([NUM_EXAMPLES, 1]))

        cxt.layer0 = tc.ml.nn.DNNLayer.create('layer0', 1, 1, tc.ml.Sigmoid())
        cxt.neural_net = tc.ml.nn.Sequential.load([cxt.layer0])

        self.execute(cxt)

    def testAnd(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES * 2).reshape([NUM_EXAMPLES, 2])
        cxt.inputs = load(inputs)

        labels = np.logical_and(inputs[:, 0] > 0.5, inputs[:, 1] > 0.5).astype(np.float32).reshape([NUM_EXAMPLES, 1])
        cxt.labels = load(labels)

        cxt.layer0 = tc.ml.nn.DNNLayer.create('layer0', 2, 1, tc.ml.ReLU())
        cxt.neural_net = tc.ml.nn.Sequential.load([cxt.layer0])

        self.execute(cxt)

    def testOr(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES * 2).reshape([NUM_EXAMPLES, 2])
        cxt.inputs = load(inputs)

        labels = np.logical_or(inputs[:, 0] > 0.5, inputs[:, 1] > 0.5).astype(np.float32).reshape([NUM_EXAMPLES, 1])
        cxt.labels = load(labels)

        cxt.layer0 = tc.ml.nn.DNNLayer.create('layer0', 2, 1, tc.ml.ReLU())
        cxt.neural_net = tc.ml.nn.Sequential.load([cxt.layer0])

        self.execute(cxt)

    def testXor_2layer(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES * 3).reshape([NUM_EXAMPLES, 3])
        cxt.inputs = load(inputs)

        labels = np.logical_xor(inputs[:, 0] > 0.5, inputs[:, 1] > 0.5).astype(np.float32).reshape([NUM_EXAMPLES, 1])
        cxt.labels = load(labels)

        cxt.layer0 = tc.ml.nn.DNNLayer.create('layer0', 3, 2, tc.ml.Sigmoid())
        cxt.layer1 = tc.ml.nn.DNNLayer.create('layer1', 2, 1, tc.ml.Sigmoid())
        cxt.neural_net = tc.ml.nn.Sequential.load([cxt.layer0, cxt.layer1])

        self.execute(cxt)

    def testXor_3layer(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES * 3).reshape([NUM_EXAMPLES, 3])
        cxt.inputs = load(inputs)

        labels = np.logical_and(inputs[:, 0] > 0.5, inputs[:, 1] < 0.5, inputs[:, 2] > 0.7).astype(np.float32)
        labels = labels.reshape([NUM_EXAMPLES, 1])
        cxt.labels = load(labels, tc.F32)

        cxt.layer0 = tc.ml.nn.DNNLayer.create('layer0', 3, 2)
        cxt.layer1 = tc.ml.nn.DNNLayer.create('layer1', 2, 2, tc.ml.ReLU())
        cxt.layer2 = tc.ml.nn.DNNLayer.create('layer2', 2, 1, tc.ml.Sigmoid())
        cxt.neural_net = tc.ml.nn.Sequential.load([cxt.layer0, cxt.layer1, cxt.layer2])

        self.execute(cxt)

    def execute(self, cxt):
        def cost(output, labels, dl=False):
            if dl:
                return output.sub(labels).mul(2).mean()
            return output.sub(labels).pow(2).mean()

        @tc.closure(cxt.labels)
        @tc.post_op
        def train_while(i: tc.UInt, output: tc.tensor.Dense):
            return (i <= MAX_ITERATIONS).logical_and(((output > 0.5) != cxt.labels).any())

        cxt.optimizer = tc.ml.optimizer.Adam.create(param_list=cxt.neural_net.get_param_list(), lr=LEARNING_RATE)
        cxt.result = tc.ml.optimizer.train(cxt.neural_net, cxt.optimizer, cxt.inputs, cxt.labels, cost, train_while)
        response = self.host.post(ENDPOINT, cxt)

        self.assertEqual(response["i"], MAX_ITERATIONS + 1)


# TODO: migrate these tests to use a hosted training backend, when available
class CNNTests(ClientTest):
    def test1D(self):
        cxt = tc.Context()

        inputs = np.arange(12*NUM_EXAMPLES).reshape((NUM_EXAMPLES, 4, 3, 1))
        cxt.inputs = load(inputs)
        labels = np.expand_dims(inputs.sum(1), 0)
        cxt.labels = load(labels)

        padding, stride = 0, 1
        cxt.layer0 = tc.ml.nn.ConvLayer.create('layer0', inputs.shape[1:], (1, 1, 1), padding=padding, stride=stride)
        cxt.neural_net = tc.ml.nn.Sequential.load([cxt.layer0])

        self.execute(cxt)

    def test2D(self):
        cxt = tc.Context()

        inputs = np.random.rand(3*5*5*NUM_EXAMPLES).reshape((NUM_EXAMPLES, 3, 5, 5))
        cxt.inputs = load(inputs)
        labels = np.expand_dims(inputs.sum(axis=1), 1)
        cxt.labels = load(labels)

        padding, stride = 1, 1
        cxt.layer0 = tc.ml.nn.ConvLayer.create('layer0', inputs.shape[1:], (1, 3, 3), padding=padding, stride=stride)
        cxt.neural_net = tc.ml.nn.Sequential.load([cxt.layer0])

        self.execute(cxt)

    @unittest.skip  # TODO: determine why this uses so much memory
    def test_combination(self):
        cxt = tc.Context()

        inputs = np.random.rand(3*5*5*NUM_EXAMPLES).reshape((NUM_EXAMPLES, 3, 5, 5))
        cxt.inputs = load(inputs)
        labels = np.expand_dims(inputs.sum(axis=1), 1)
        cxt.labels = load(labels)

        padding, stride = 1, 1
        cxt.layer0 = tc.ml.nn.ConvLayer.create('layer0', inputs.shape[1:], (2, 1, 1), padding=padding, stride=stride)
        cxt.layer1 = tc.ml.nn.ConvLayer.create('layer1', (2, 5, 5), (1, 3, 3), padding=padding, stride=stride)
        cxt.neural_net = tc.ml.nn.Sequential.load([cxt.layer0, cxt.layer1])

        self.execute(cxt)

    def execute(self, cxt):
        def cost(output, labels, dl=False):
            if dl:
                return output.sub(labels).mul(2).mean()
            return output.sub(labels).pow(2).mean()

        @tc.closure(cxt.labels)
        @tc.post_op
        def train_while(i: tc.UInt, loss):
            return (i <= MAX_ITERATIONS).logical_and(tc.F32(loss) > 0.5)

        cxt.optimizer = tc.ml.optimizer.Adam.create(param_list=cxt.neural_net.get_param_list(), lr=LEARNING_RATE)
        cxt.result = tc.ml.optimizer.train(cxt.neural_net, cxt.optimizer, cxt.inputs, cxt.labels, cost, train_while)

        response = self.host.post(ENDPOINT, cxt)

        self.assertEqual(response["i"], MAX_ITERATIONS + 1)


def load(ndarray, dtype=tc.F32):
    return tc.tensor.Dense.load(ndarray.shape, ndarray.flatten().tolist(), dtype)


if __name__ == "__main__":
    unittest.main()
