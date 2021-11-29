import math

import numpy as np
import operator
import tinychain as tc
import unittest

from functools import reduce
from testutils import ClientTest

Dense = tc.tensor.Dense


ENDPOINT = "/transact/hypothetical"
LEARNING_RATE = 0.01
MAX_ITERATIONS = 1000


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
@unittest.skip
class DNNTests(ClientTest):
    @staticmethod
    def create_layer(input_size, output_size, activation):
        shape = (input_size, output_size)
        bias = tc.tensor.Dense.load([output_size], tc.F32, truncated_normal(output_size).tolist())
        weights = tc.tensor.Dense.load(shape, tc.F32, truncated_normal(input_size * output_size).tolist())
        return tc.ml.dnn.layer(weights, bias, activation)

    def testNot(self):
        cxt = tc.Context()

        cxt.inputs = Dense.load([2, 1], tc.Bool, [True, False])
        cxt.labels = Dense.load([2, 1], tc.Bool, [False, True])

        cxt.input_layer = self.create_layer(1, 1, tc.ml.Sigmoid())
        cxt.nn = tc.ml.dnn.neural_net([cxt.input_layer])

        self.execute(cxt)

    def testIdentity(self):
        cxt = tc.Context()

        cxt.inputs = Dense.load([2, 1], tc.Bool, [True, False])
        cxt.labels = Dense.load([2, 1], tc.Bool, [True, False])

        cxt.input_layer = self.create_layer(1, 1, tc.ml.Sigmoid())
        cxt.nn = tc.ml.dnn.neural_net([cxt.input_layer])

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

        cxt.input_layer = self.create_layer(2, 1, tc.ml.ReLU())
        cxt.nn = tc.ml.dnn.neural_net([cxt.input_layer])

        self.execute(cxt)

    def execute(self, cxt):
        @tc.closure
        @tc.post_op
        def while_cond(output: Dense, i: tc.UInt):
            fit = ((output > 0.5) != cxt.labels).any()
            return fit.logical_and(i < MAX_ITERATIONS)
        
        cxt.optimizer = tc.ml.Adam_Optimizer.create()
        @tc.closure
        @tc.post_op
        def train(i: tc.UInt):          
            output,optimizer = cxt.nn.train(cxt.inputs, lambda output: ((output - cxt.labels)**2) * LEARNING_RATE , cxt.optimizer)
            return {"output": output, "optimizer":optimizer,"i": i + 1}

        cxt.result = tc.While(while_cond, train, {"output": cxt.nn.eval(cxt.inputs),"optimizer":cxt.optimizer,"i": 0})

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

        return tc.ml.cnn.conv_1d(input_shape, weights, bias, stride, activation)

    @classmethod
    def conv_2d(cls, input_shape, output_shape, kernel_shape, stride=1, activation=tc.ml.Sigmoid()):
        assert len(input_shape) == 3
        assert len(output_shape) == 3

        weights, bias = cls.conv(2, input_shape, output_shape, kernel_shape)

        return tc.ml.cnn.conv_2d(input_shape, weights, bias, stride, activation)

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
        cxt.result = cxt.layer.eval(cxt.inputs)

        response = self.host.post(ENDPOINT, cxt)
        print(response)


if __name__ == "__main__":
    unittest.main()
