# Constructors for a generic deep neural network.
#
# Prefer this implementation if no more domain-specific neural net architecture is needed.

from tinychain.collection.tensor import einsum, Dense
from tinychain.ml import Layer, NeuralNet, Sigmoid, Parameter
from tinychain.value import String

class DNNLayer(Layer):
    @classmethod
    def create(cls, name: str, input_size, output_size, activation=Sigmoid()):
        """Create a new, empty `DNNLayer` with the given shape and activation function."""

        weights = Dense.create((input_size, output_size))
        bias = Dense.create((output_size,))
        name = String(name + '.{{nv}}')

        return cls.load(name, weights, bias, activation)

    @classmethod
    def load(cls, name, weights, bias, activation=Sigmoid()):
        """Load a `DNNLayer` with the given `weights` and `bias` tensors."""

        class _DNNLayer(cls):
            @classmethod
            @property
            def shape(cls):
                return {"weights": weights.shape, "bias": bias.shape}

            @property
            def activation(self):
                return activation

        return _DNNLayer({'name': name, "weights": weights, "bias": bias})

    @property
    def activation(self):
        return Sigmoid()

    def forward(self, x):
        inputs = einsum("ki,ij->kj", [x, self["weights"]]) + self["bias"]
        return self.activation.forward(inputs)

    def backward(self, x, loss):
        m = x.shape[0]
        inputs = einsum("ki,ij->kj", [x, self["weights"]]) + self["bias"]
        delta = loss * self.activation.backward(inputs)
        update = einsum("ki,kj->ij", [x, delta]).copy()
        return delta, [
            Parameter(name=self['name'].render(nv='weight'), value=self["weights"], grad=update / m),
            Parameter(name=self['name'].render(nv='bias'), value=self["bias"], grad=delta.sum(0) / m)
            ]


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
            @property
            def shape(cls):
                return [layer.shape for layer in layers]

            def forward(self, inputs):
                state = self[0].forward(inputs)
                for i in range(1, n):
                    state = self[i].forward(state)

                return state

            def backward(self, inputs, loss):
                layer_inputs = [inputs]

                for l in range(n):
                    layer_output = self[l].forward(layer_inputs[-1]).copy()
                    layer_inputs.append(layer_output)

                param_list = []
                for l in reversed(range(n)):
                    loss, layer_param_list = self[l].backward(layer_inputs[l], loss)
                    loss = loss.copy()
                    param_list.extend(layer_param_list)

                return param_list

        return DNN(layers)