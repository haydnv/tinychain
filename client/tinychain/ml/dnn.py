# Constructors for a generic deep neural network.
#
# Prefer this implementation if no more domain-specific neural net architecture is needed.

from tinychain.collection.tensor import einsum, Dense
from tinychain.error import BadRequest
from tinychain.ml import Gradient, Layer, NeuralNet, Sigmoid
from tinychain.ref import After, If
from tinychain.state import Map, Tuple
from tinychain.value import Bool, String


class DNNLayer(Layer):
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
            @classmethod
            def shape(cls):
                return {"weights": weights.shape(), "bias": bias.shape()}

            @property
            def activation(self):
                return activation

        return _DNNLayer({"weights": weights, "bias": bias})

    @property
    def activation(self):
        return Sigmoid()

    def eval(self, inputs):
        return self.activation.forward(einsum("ij,ki->kj", [self["weights"], inputs])) + self["bias"]

    def train(self, i, inputs, loss, optimizer):
        dZ = self.activation.backward(loss, einsum("ij,ki->kj", [self["weights"], inputs]))

        weight_gradients = einsum("kj,ki->ij", [dZ, inputs])
        bias_gradients = dZ.sum(0)

        delta = optimizer.optimize(i, {"weights": weight_gradients, "bias": bias_gradients})
        update = self.write((self["weights"] - delta["weights"]), (self["bias"] - delta["bias"]))
        return After(update, dZ)

    def write(self, weights, bias):
        """Overwrite the weights and bias of this layer."""

        return self["weights"].write(weights), self["bias"].write(bias)


class DNN(NeuralNet):
    @classmethod
    def create(cls, shape):
        """Create a new, zero-values multilayer deep neural network (DNN).

        Args:
            `shape` a list of tuples of the form `input_size, output_size` or `input_size, output_size, activation`
        """

        layers = [DNNLayer.create(*ioa) for ioa in shape]
        return cls.load(layers)

    @classmethod
    def load(cls, layers):
        if not layers:
            raise ValueError("cannot initialize a neural net with no layers")

        n = len(layers)

        class DNN(cls):
            @classmethod
            def shape(cls):
                return [layer.shape() for layer in layers]

            def eval(self, inputs):
                state = self[0].eval(inputs)
                for i in range(1, n):
                    state = self[i].eval(state)

                return state

            def train(self, i, inputs, loss, optimizer):
                layer_inputs = [inputs]
                for l in range(n):
                    layer_inputs.append(self[l].eval(layer_inputs[-1]))

                for l in reversed(range(n)):
                    loss = self[l].train(i, layer_inputs[l], loss, optimizer)

                return loss

            def write(self, layers):
                updates = []
                for i in range(n):
                    w, b = Tuple(layers[i]).unpack(2)
                    updates.append(self[i].write(w, b))

                err_msg = (String("DNN.write expected {{exp}} layers but found {{act}}")
                           .render(exp=n, act=layers.len()))

                return If(layers.len() == n, updates, BadRequest(err_msg))

        return DNN(layers)

    def write(self, layers):
        """Overwrite the weights and biases of the layers of this neural net."""

        raise NotImplementedError("use DNN.create or DNN.load to initialize a new deep neural net")
