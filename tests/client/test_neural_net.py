import math
import numpy as np
import operator
import tinychain as tc
import unittest

from functools import reduce
from testutils import ClientTest

Dense = tc.tensor.Dense


ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = tc.F32(1)
MAX_ITERATIONS = 500
NUM_EXAMPLES = 20


def truncated_normal(size, mean=0., std=None):
    std = std if std else math.sqrt(size)

    while True:
        dist = np.random.normal(mean, std, size)
        truncate = np.abs(dist) > mean + (std * 2)
        if truncate.any():
            new_dist = np.random.normal(mean, std, size) * truncate
            dist *= np.logical_not(truncate)
            dist += new_dist
        else:
            return dist


# TODO: implement AdamOptimizer
class DNNTests(ClientTest):
    @classmethod
    def setUpClass(cls):
        np.random.seed()
        super().setUpClass()

    @staticmethod
    def create_layer(name, input_size, output_size, activation):
        shape = (input_size, output_size)
        bias = tc.tensor.Dense.load([output_size], tc.F32, truncated_normal(output_size).tolist())
        weights = tc.tensor.Dense.load(shape, tc.F32, truncated_normal(input_size * output_size).tolist())
        return tc.ml.dnn.DNNLayer.load(name, weights, bias, activation)

    def testNot(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES).reshape([NUM_EXAMPLES, 1])
        cxt.inputs = load(inputs)
        cxt.labels = load(inputs < 0.5, tc.Bool)

        cxt.input_layer = self.create_layer(tc.String('layer0'), 1, 1, tc.ml.Sigmoid())
        cxt.nn = tc.ml.dnn.DNN.load([cxt.input_layer])

        self.execute(cxt)

    def testAnd(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES * 2).reshape([NUM_EXAMPLES, 2])
        cxt.inputs = load(inputs)
        cxt.labels = load(np.logical_and(inputs[:, 0] > 0.5, inputs[:, 1] > 0.5), tc.Bool)

        cxt.input_layer = self.create_layer(tc.String('layer0'), 2, 1, tc.ml.ReLU())
        cxt.nn = tc.ml.dnn.DNN.load([cxt.input_layer])

        self.execute(cxt)

    def testOr(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES * 2).reshape([NUM_EXAMPLES, 2])
        cxt.inputs = load(inputs)
        cxt.labels = load(np.logical_or(inputs[:, 0] > 0.5, inputs[:, 1] > 0.5), tc.Bool)

        cxt.input_layer = self.create_layer(tc.String('layer0'), 2, 1, tc.ml.ReLU())
        cxt.nn = tc.ml.dnn.DNN.load([cxt.input_layer])

        self.execute(cxt)

    def testXor(self):
        cxt = tc.Context()

        inputs = np.random.random(NUM_EXAMPLES * 5).reshape([NUM_EXAMPLES, 5])
        labels = np.logical_xor(inputs[:, 0] > 0.5, inputs[:, 1] > 0.5).reshape([NUM_EXAMPLES, 1])
        cxt.inputs = load(inputs)
        cxt.labels = load(labels, tc.Bool)

        cxt.input_layer = self.create_layer(tc.String('layer0'), 5, 3, tc.ml.Sigmoid())
        cxt.input_layer2 = self.create_layer(tc.String('layer1'), 3, 2, tc.ml.Sigmoid())
        cxt.output_layer = self.create_layer(tc.String('layer2'), 2, 1, tc.ml.ReLU())
        cxt.nn = tc.ml.dnn.DNN.load([cxt.input_layer, cxt.input_layer2, cxt.output_layer])

        self.execute(cxt)

    def execute(self, cxt):
        def cost(output):
            return (output - cxt.labels)**2

        cxt.optimizer = tc.ml.optimizer.GradientDescent.create(LEARNING_RATE)
        cxt.result = tc.ml.optimizer.train(cxt.nn, cxt.optimizer, cxt.inputs, cost, MAX_ITERATIONS)

        response = self.host.post(ENDPOINT, cxt)
        self.assertLess(response["i"], MAX_ITERATIONS, "failed to converge")

    def testAdam(self):
        cxt = tc.Context()

        def BCELoss(output):
            return cxt.lab*output.log() - (cxt.lab.add(-1)) * (output.add(-1)*-1).log()

        inputs = np.random.random(50).reshape([10, 5])
        lab = np.logical_xor(inputs[:, 0] > 0.5, inputs[:, 1] > 0.5).reshape([10, 1]).astype(np.float32)
        cxt.inputs = load(inputs)
        cxt.lab = load(lab, tc.F32)

        cxt.layer0 = self.create_layer(tc.String('layer0'), 5, 3, tc.ml.Sigmoid())
        cxt.layer1 = self.create_layer(tc.String('layer1'), 3, 2, tc.ml.ReLU())
        cxt.layer2 = self.create_layer(tc.String('layer2'), 2, 1, tc.ml.Sigmoid())
        cxt.model = tc.ml.dnn.DNN.load([cxt.layer0, cxt.layer1, cxt.layer2])
        cxt.optimizer = tc.ml.optimizer.Adam.create(lr=tc.F32(1e-2))
        cxt.result = tc.ml.optimizer.train(cxt.model, cxt.optimizer, cxt.lab, BCELoss, MAX_ITERATIONS)
        response = self.host.post(ENDPOINT, cxt)
        self.assertLess(response["i"], MAX_ITERATIONS, "failed to converge")


# TODO: implement AdamOptimizer
class CNNTests(ClientTest):
    @classmethod
    def conv(cls, num_spatial_dims, input_shape, output_shape, kernel_shape):
        input_channels = input_shape[-1]
        output_channels = output_shape[-1]

        weight_shape = kernel_shape * num_spatial_dims
        weight_shape += (input_channels, output_channels)

        bias = tc.tensor.Dense.zeros([output_channels])

        weights = truncated_normal(reduce(operator.mul, weight_shape))
        weights = tc.tensor.Dense.load(weight_shape, tc.F32, weights.tolist())

        return weights, bias

    @classmethod
    def conv_1d(cls, input_shape, output_shape, kernel_shape, stride=1, activation=tc.ml.Sigmoid()):
        assert len(input_shape) == 2
        assert len(output_shape) == 2

        weights, bias = cls.conv(1, input_shape, output_shape, kernel_shape)

        return tc.ml.cnn.Conv1D.load(input_shape, weights, bias, stride, activation)

    @classmethod
    def conv_2d(cls, input_shape, output_shape, kernel_shape, stride=1, activation=tc.ml.Sigmoid()):
        assert len(input_shape) == 3
        assert len(output_shape) == 3

        weights, bias = cls.conv(2, input_shape, output_shape, kernel_shape)

        return tc.ml.cnn.Conv2D.load(input_shape, weights, bias, stride, activation)

    def test1D(self):
        data = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]

        inputs = np.expand_dims(np.array(data), -1)
        labels = np.expand_dims(np.array(data), -1)

        self.execute(inputs, labels, self.conv_1d)

    def test2D(self):
        none = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        horiz = [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ]

        vert = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]

        cross = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]

        inputs = np.expand_dims(np.array([none, horiz, vert, cross]), -1)
        labels = np.expand_dims(np.array([none, horiz, vert, cross]), -1)

        self.execute(inputs, labels, self.conv_2d)

    def execute(self, inputs, labels, conv):
        cxt = tc.Context()
        cxt.inputs = tc.tensor.Dense.load(inputs.shape, tc.F32, inputs.flatten().tolist())
        cxt.labels = tc.tensor.Dense.load(labels.shape, tc.F32, labels.flatten().tolist())
        cxt.layer = conv(inputs.shape[1:], labels.shape[1:], [2], 2)
        cxt.result = cxt.layer.forward(cxt.inputs)

        response = self.host.post(ENDPOINT, cxt)
        print(response)


def load(ndarray, dtype=tc.F32):
    return tc.tensor.Dense.load(ndarray.shape, dtype, ndarray.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
