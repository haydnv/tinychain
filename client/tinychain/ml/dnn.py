# Constructors for a generic deep neural network.
#
# Prefer this implementation if no more domain-specific neural net architecture is needed.
import functools
import operator
from typing import List

from tinychain.collection.tensor import einsum, Dense, Tensor
from tinychain.ml import Layer, NeuralNet, Sigmoid, DiffedParameter, Parameter


class DNNLayer(Layer):
    @classmethod
    def create(cls, name: str, input_size, output_size, activation=Sigmoid()):
        """Create a new, empty `DNNLayer` with the given shape and activation function."""

        weights = Dense.create((input_size, output_size))
        bias = Dense.create((output_size,))

        return cls.load(name, weights, bias, activation)

    @classmethod
    def load(cls, name, weights, bias, activation=Sigmoid()):
        """Load a `DNNLayer` with the given `weights` and `bias` tensors."""

        class _DNNLayer(cls):
            @classmethod
            @property
            def shape(cls):
                return {name + ".weights": weights.shape, name + ".bias": bias.shape}

            def forward(self, x):
                inputs = einsum("ki,ij->kj", [x, self[name + ".weights"]]) + self[name + ".bias"]
                return activation.forward(inputs)

            def backward(self, x, loss):
                m = x.shape[0]
                inputs = einsum("ki,ij->kj", [x, self[name + ".weights"]]) + self[name + ".bias"]
                delta = Tensor(activation.backward(inputs) * loss)
                dL = einsum("ij,kj->ki", [self[name + ".weights"], delta])
                return dL, [
                    DiffedParameter.create(
                        name=name + '.weight',
                        value=self[name + ".weights"],
                        grad=einsum("ki,kj->ij", [x, delta]).copy() / m),
                    DiffedParameter.create(
                        name=name + '.bias',
                        value=self[name + ".bias"],
                        grad=delta.sum(0) / m)
                ]

            def get_param_list(self) -> List[Parameter]: 
                return [
                    Parameter.create(name=name + '.weight', value=self[name + ".weights"]),
                    Parameter.create(name=name + '.bias', value=self[name + ".bias"])
                ]

        return _DNNLayer({name + ".bias": bias, name + ".weights": weights})


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

                return loss, param_list
            
            def get_param_list(self)  -> List[Parameter]:
                return functools.reduce(operator.add, [layer.get_param_list() for layer in layers], [])

        return DNN(layers)