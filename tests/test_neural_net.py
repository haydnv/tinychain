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
    def eval(self, cxt, state: tc.Map) -> tc.Map:
        cxt.dot = Dense(state["Z"]) * self.weights.transpose()
        cxt.Z = (cxt.dot + self.bias).sum(1)
        cxt.A = self.activation.forward(Z=cxt.Z)
        return {"A": cxt.A, "Z": cxt.Z}


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
        inner = Layer(self[0]).eval(state=tc.Map(Z=inputs.expand_dims(-1)))
        output = Layer(self[1]).eval(state=inner)
        return output["Z"]

    def error(self, inputs, labels):
        output = self.eval(inputs)
        return (output - labels)**2

    def train(self, inputs, labels):
        num_layers = 2  # TODO: parameterize this by putting it into a schema class

        outer_layer = Layer(self[1])
        inner_layer = Layer(self[0])

        inner = inner_layer.eval(state=tc.Map(Z=inputs.expand_dims()))
        output = outer_layer.eval(state=inner)
        error = (Dense(output["Z"]) - labels)**2

        dA = (outer_layer.weights / error) - ((1 - outer_layer.weights) / (1 - error))
        dZ = outer_layer.activation.backward(dA=dA, Z=output["Z"]).copy()
        d_weights = (dZ.transpose() * inner["A"]).sum(0) / num_layers
        d_bias = dZ.sum(1) / num_layers

        update_outer = (
            outer_layer.weights.write(None, outer_layer.weights - (LEARNING_RATE * d_weights)),
            outer_layer.bias.write(None, outer_layer.bias - (LEARNING_RATE * d_bias)),
        )

        dA = tc.tensor.einsum("ji,jk->kj", [outer_layer.weights, dZ])
        dZ = inner_layer.activation.backward(dA=dA, Z=inner["Z"])
        d_weights = (dZ * inner["A"]).sum(0) / num_layers
        d_bias = dZ.sum(0) / num_layers

        update_inner = (
            inner_layer.weights.write(None, inner_layer.weights - (LEARNING_RATE * d_weights)),
            inner_layer.bias.write(None, inner_layer.bias - (LEARNING_RATE * d_bias)),
        )

        return update_outer, update_inner


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

        cxt.result = tc.After(cxt.nn.train(cxt.inputs, cxt.labels), cxt.nn.eval(cxt.inputs))

        response = self.host.post(ENDPOINT, cxt)

        contents = response[str(tc.uri(Dense))]
        self.assertEqual(contents[0], [[4], str(tc.uri(tc.F32))])
        self.assertEqual(len(contents[1]), 4)

    @classmethod
    def tearDownClass(cls):
        cls.host.stop()


if __name__ == "__main__":
    unittest.main()
