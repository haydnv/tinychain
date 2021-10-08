from tinychain.collection.tensor import einsum
from tinychain.state import Tuple


def layer(weights):
    class Layer(Tuple):
        def eval(self, inputs):
            return einsum("ij,ki->kj", [weights, inputs])

    return Layer(weights)


def neural_net(layers):
    class NeuralNet(Tuple):
        def eval(self, inputs):
            state = layers[0].eval(inputs)
            for i in range(1, len(layers)):
                state = layers[i].eval(state)

            return state

    return NeuralNet(layers)
