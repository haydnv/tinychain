# Constructors for a generic deep neural network.
#
# Prefer this implementation if no more domain-specific neural net architecture is needed.

from tinychain.collection.tensor import einsum, Dense
from tinychain.ml import Layer, NeuralNet, Sigmoid
from tinychain.ref import After
from tinychain.state import Map


class DNNLayer(Layer):
    ERR_BUILDER = "Use DNNLayer.create to construct a new DNNLayer"

    @classmethod
    def create(cls, input_size, output_size, activation=Sigmoid()):
        """Create a new, empty `DNNLayer` with the given shape and activation function."""

        weights = Dense.create((input_size, output_size))
        bias = Dense.create((output_size,))

        return cls.load(weights, bias, activation)

    @classmethod
    def load(cls, weights, bias, activation=Sigmoid()):
        """Load a `DNNLayer` with the given `weights` and `bias` tensors."""

        class _DNNLayer(cls):
            def eval(self, inputs):
                return activation.forward(einsum("ij,ki->kj", [self["weights"], inputs])) + self["bias"]

            def gradients(self, A_prev, dA, Z):
                dZ = activation.backward(dA, Z).copy()
                dA_prev = einsum("kj,ij->ki", [dZ, self["weights"]])
                d_weights = einsum("kj,ki->ij", [dZ, A_prev])
                d_bias = dZ.sum(0)
                return dA_prev, d_weights, d_bias

            def train_eval(self, inputs):
                Z = einsum("ij,ki->kj", [self["weights"], inputs])
                A = activation.forward(Z) + self["bias"]
                return A, Z

            def update(self, d_weights, d_bias):
                return self["weights"].write(self["weights"] - d_weights), self["bias"].write(self["bias"] - d_bias)

            def write(self, weights, bias):
                return self["weights"].write(weights), self["bias"].write(bias)

        return _DNNLayer(Map(weights=weights, bias=bias))

    def eval(self, inputs):
        raise NotImplementedError(self.ERR_BUILDER)

    def train_eval(self, inputs):
        raise NotImplementedError(self.ERR_BUILDER)


class DNN(NeuralNet):
    @classmethod
    def create(cls, shape):
        """Create a new, zero-values multilayer deep neural network (DNN).

        Args:
            `shape` a list of tuples of the form `input_size, output_size` or `input_size, output_size, activation`
        """

        layers = [Layer.create(*ioa) for ioa in shape]
        return cls.load(layers)

    @classmethod
    def load(cls, layers):
        class DNN(cls):
            def eval(self, inputs):
                state = layers[0].eval(inputs)
                for i in range(1, len(layers)):
                    state = layers[i].eval(state)

                return state

            def train(self, inputs, cost):
                A = [inputs]
                Z = [None]

                for layer in layers:
                    A_l, Z_l = layer.train_eval(A[-1])
                    A.append(A_l.copy())
                    Z.append(Z_l)

                m = inputs.shape[0]
                dA = cost(A[-1]).sum() / m

                updates = []
                for i in reversed(range(0, len(layers))):
                    dA, d_weights, d_bias = layers[i].gradients(A[i], dA, Z[i + 1])
                    update = layers[i].update(d_weights, d_bias)
                    updates.append(update)

                return After(updates, A[-1])

        return DNN(layers)
