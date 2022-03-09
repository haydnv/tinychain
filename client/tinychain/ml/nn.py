import functools
import operator

from typing import List

from ..collection.tensor import Dense, Tensor, einsum
from ..ml import DiffedParameter, Identity, Layer, NeuralNet, Parameter
from ..scalar.ref import After
from ..scalar.number import Int


class DNNLayer(Layer):
    @classmethod
    def create(cls, name: str, input_size, output_size, activation=Identity()):
        """Create a new, empty `DNNLayer` with the given shape and activation function. Initializing `ConvLayer`
        parameters by Xavier initialization for Sigmoid, Tanh activations, and Kaiming initialization for ReLU.

        Args:
            `name`: name and number of layer;
            `input_size`: size of inputs;
            `output_size`: size of outputs;
            `activation`: activation function.
        """

        std = activation.optimal_std(input_size, output_size)
        weights = Dense.random_normal(shape=[input_size, output_size], mean=0.0, std=std)
        bias = Dense.random_normal(shape=[output_size, ], mean=0.0, std=std)

        return cls.load(name, weights, bias, activation)

    @classmethod
    def load(cls, name, weights, bias, activation):
        """Load a `DNNLayer` with the given `weights` and `bias` tensors."""

        class _DNNLayer(cls):
            @classmethod
            @property
            def shape(cls):
                return weights.shape

            def forward(self, x):
                inputs = Tensor(einsum("ki,ij->kj", [x, self[name + ".weights"]])) + self[name + ".bias"]
                return activation.forward(inputs), None

            def backward(self, x, loss):
                m = Int(x.shape[0])
                inputs = einsum("ki,ij->kj", [x, self[name + ".weights"]]) + self[name + ".bias"]
                delta = Tensor(activation.backward(inputs) * loss)
                dL = einsum("ij,kj->ki", [self[name + ".weights"], delta])
                return dL, [
                    DiffedParameter.create(
                        name=name + '.weights',
                        value=Tensor(self[name + ".weights"]),
                        grad=einsum("ki,kj->ij", [x, delta]).copy() / m),
                    DiffedParameter.create(
                        name=name + '.bias',
                        value=Tensor(self[name + ".bias"]),
                        grad=delta.sum(0) / m)
                ]

            def get_param_list(self) -> List[Parameter]:
                return [
                    Parameter.create(name=name + '.weights', value=Tensor(self[name + ".weights"])),
                    Parameter.create(name=name + '.bias', value=Tensor(self[name + ".bias"]))
                ]

        return _DNNLayer({name + ".bias": bias, name + ".weights": weights})


class ConvLayer(Layer):
    @classmethod
    def create(cls, name: str, inputs_shape, filter_shape, stride=1, padding=1, activation=Identity()):
        """Create a new, empty `ConvLayer` with the given shape and activation function. Initializing `ConvLayer`
        parameters by Xavier initialization for Sigmoid, Tanh activations, and Kaiming initialization for ReLU

        Args:
            `name`: name and number of layer;
            `inputs_stape`: size of inputs [c_i, h_i, w_i] where
                `c_i`: number of channels;
                `h_i`: height's width  for each channel;
                'w_i': matrix's width  for each channel.
            `filter_shape`: size of filter [h, w, out_c] where
                `out_c`: number of output channels;
                `h_f`: height of the kernel;
                'w_f`: width of the kernel.
            `activation`: activation function.
        """

        c_i, h_i, w_i = inputs_shape
        out_c, h_f, w_f = filter_shape

        std = activation.optimal_std(c_i * h_i * w_i, out_c * h_f * w_f)
        weights = Dense.random_normal([out_c, c_i, h_f, w_f], mean=0.0, std=std)
        bias = Dense.random_normal([out_c, 1], mean=0.0, std=std)

        return cls.load(name, weights, bias, inputs_shape, filter_shape, stride, padding, activation)

    @classmethod
    def load(cls, name, weights, bias, inputs_shape, filter_shape, stride, padding, activation):
        """Load a `ConvLayer` with the given `weights` and `bias` tensors."""
        class _ConvLayer(cls):
            @classmethod
            @property
            def shape(cls):
                return weights.shape

            def forward(self, x: Dense):
                c_i, h_i, w_i = inputs_shape
                out_c, h_f, w_f = filter_shape
                b_i = x.shape[0]
                h_out = int((h_i - h_f + 2 * padding) / stride + 1)
                w_out = int((w_i - w_f + 2 * padding) / stride + 1)
                pad_matrix = Dense.zeros([b_i, c_i, h_i + padding * 2, w_i + padding * 2])
                im2col_matrix = []
                for i in range(h_out):
                    for j in range(w_out):
                        im2col_matrix.append(pad_matrix[:, :, i:i + h_f, j:j + w_f].reshape([c_i * h_f * w_f, b_i]))
                im2col_concat = Tensor(After(pad_matrix[:, :, padding:(padding + h_i), padding:(padding + w_i)].write(x.copy()), Dense.concatenate(im2col_matrix, 0)))
                im2col_matrix = Tensor(After(im2col_concat, im2col_concat.reshape([b_i * h_out * w_out, c_i * h_f * w_f]).transpose()))
                w_col = self[name + '.weights'].reshape([out_c, c_i * h_f * w_f])
                in2col_multiply = Tensor(einsum("ij,jm->im", [w_col, im2col_matrix]) + self[name + '.bias']).reshape([out_c, h_out, w_out, b_i])
                output = Tensor(in2col_multiply.copy().transpose([3, 0, 1, 2]))

                return activation.forward(output), im2col_matrix

            def backward(self, x, loss):
                c_i, h_i, w_i = inputs_shape
                out_c, h_f, w_f = filter_shape
                b_i = x.shape[0]
                h_out = int((h_i - h_f + 2 * padding) / stride + 1)
                w_out = int((w_i - w_f + 2 * padding) / stride + 1)
                inputs, im2col_matrix = self.forward(x)
                delta = Tensor(activation.backward(Tensor(inputs)) * loss)
                delta_reshaped = Tensor(delta.transpose([1, 2, 3, 0])).reshape([out_c, h_out * w_out * b_i])
                dw = Tensor(einsum('ij,mj->im', [delta_reshaped, im2col_matrix])).reshape(self[name + '.weights'].shape)
                db = Tensor(einsum('ijkb->j', [delta])).reshape([out_c, 1])
                dloss_col = Tensor(einsum('ji,jm->im', [self[name + '.weights'].reshape([out_c, c_i * h_f * w_f]), delta_reshaped]))
                dloss_col_reshaped = dloss_col.reshape([c_i, h_f, w_f, h_out, w_out, b_i]).copy().transpose([5, 0, 3, 4, 1, 2])

                # TODO: make this a property of the ConvLayer instance
                pad_matrix = Dense.zeros([b_i, c_i, h_i + padding * 2, w_i + padding * 2])

                result = [
                    pad_matrix[:, :, i:i + h_f, j:j + w_f].write(pad_matrix[:, :, i:i + h_f, j:j + w_f].copy() + dloss_col_reshaped[:, :, i, j, :, :])
                    for i in range(h_out) for j in range(w_out)
                    ]
                dloss_result = Tensor(After(result, pad_matrix[:, :, padding:(padding + h_i), padding:(padding + w_i)]))

                return dloss_result, [
                    DiffedParameter.create(
                        name=name + '.weights',
                        value=self[name + ".weights"],
                        grad=dw),
                    DiffedParameter.create(
                        name=name + '.bias',
                        value=self[name + ".bias"],
                        grad=db)
                ]

            def get_param_list(self) -> List[Parameter]:
                return [
                    Parameter.create(name=name + '.weights', value=self[name + ".weights"]),
                    Parameter.create(name=name + '.bias', value=self[name + ".bias"])
                ]

        return _ConvLayer({name + ".weights": weights, name + ".bias": bias})


class Sequential(NeuralNet):
    """Create a new NeuralNet as list `Layer`'s. `Layer`'s could be `DNNLayer` and `ConvLayer`.
        Args:
            `layers` a list of exemplar `DNNLayer` or `ConvLayer` with parameters:
    """

    @classmethod
    def load(cls, layers):
        if not layers:
            raise ValueError("cannot initialize a neural net with no layers")

        n = len(layers)

        class Sequential(cls):
            @classmethod
            @property
            def shape(cls):
                return [layer.shape for layer in layers]

            def forward(self, inputs):
                state, _ = self[0].forward(inputs)
                for i in range(1, n):
                    state, _ = self[i].forward(state)
                return state

            def backward(self, inputs, loss):
                layer_inputs = [inputs]

                for l in range(n):
                    layer_output, _ = self[l].forward(layer_inputs[-1])
                    layer_output = layer_output.copy()
                    layer_inputs.append(layer_output)

                param_list = []
                for l in reversed(range(n)):
                    loss, layer_param_list = self[l].backward(layer_inputs[l], loss)
                    loss = loss.copy()
                    param_list.extend(layer_param_list)

                return loss, param_list

            def get_param_list(self) -> List[Parameter]:
                return functools.reduce(operator.add, [layer.get_param_list() for layer in layers], [])

        return Sequential(layers)
