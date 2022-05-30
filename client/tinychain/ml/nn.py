""":class:`NeuralNet` and :class:`Layer` :class:`Model` definitions with common implementations"""

import logging

from ..app import Dynamic, Model
from ..collection.tensor import einsum, Dense, Tensor
from ..context import deanonymize
from ..decorators import differentiable
from ..generic import Tuple
from ..math.operator import derivative_of, operator, Dual
from ..scalar.number import Float
from ..scalar.ref import After

from .constants import LIB_URI
from .interface import Differentiable
from .variable import Variable


class Layer(Model, Differentiable):
    """A :class:`Layer` in a :class:`NeuralNet`"""

    __uri__ = LIB_URI + "/Layer"


class ConvLayer(Layer, Dynamic):
    @classmethod
    def create(cls, inputs_shape, filter_shape, stride=1, padding=1, activation=None, optimal_std=None):
        """
        Create a new, empty :class:`ConvLayer` with the given shape and activation function.

        Args:
            `inputs_shape`: size of inputs `[c_i, h_i, w_i]` where
                `c_i`: number of channels,
                `h_i`: channel height,
                'w_i': channel width;
            `filter_shape`: size of filter `[h_f, w_f, out_c]` where
                `out_c`: number of output channels,
                `h_f`: kernel height,
                'w_f`: kernel width;
            `activation`: activation function
        """

        c_i, h_i, w_i = inputs_shape
        out_c, h_f, w_f = filter_shape

        input_size = c_i * h_i * w_i
        output_size = out_c * h_f * w_f

        if callable(optimal_std):
            std = optimal_std(input_size, output_size)
        elif optimal_std:
            std = optimal_std
        else:
            std = (input_size * output_size)**0.5

        weights = Variable.random_normal([out_c, c_i, h_f, w_f], mean=0.0, std=std)
        bias = Variable.random_normal([out_c, 1], mean=0.0, std=std)

        return cls(weights, bias, inputs_shape, filter_shape, stride, padding, activation)

    def __init__(self, weights, bias, inputs_shape, filter_shape, stride, padding, activation):
        if not padding or padding < 0:
            raise ValueError(f"invalid padding for ConvLayer: {padding}")

        if not isinstance(weights, Variable):
            logging.warning(f"ConvLayer with weights {weights} will not be trainable")

        if not isinstance(bias, Variable):
            logging.warning(f"ConvLayer with bias {bias} will not be trainable")

        # compile-time constants
        self._inputs_shape = inputs_shape
        self._filter_shape = filter_shape
        self._stride = stride
        self._padding = padding
        self._activation = activation

        # run-time state
        self.weights = weights
        self.bias = bias

        Dynamic.__init__(self)

    @differentiable
    def eval(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.shape[0]

        # TODO: this expectation should be set using a generic type
        inputs = Tensor.expect([batch_size] + self._inputs_shape, Float)(form=inputs)

        padding = self._padding
        stride = self._stride

        c_i, h_i, w_i = self._inputs_shape
        out_c, h_f, w_f = self._filter_shape

        h_out = int(((h_i - h_f) + (2 * padding)) / (stride + 1))
        w_out = int(((w_i - w_f) + (2 * padding)) / (stride + 1))

        assert h_out
        assert w_out

        class Convolution(Dual):
            def __init__(self, weights, inputs):
                Dual.__init__(self, weights, inputs)

                pad_matrix = Dense.zeros([batch_size, c_i, h_i + padding * 2, w_i + padding * 2])
                pad_matrix = Tensor(After(
                    pad_matrix[:, :, padding:(padding + h_i), padding:(padding + w_i)].write(self.args),
                    pad_matrix))

                shape = [c_i * h_f * w_f, batch_size]

                im2col_matrix = []
                for i in range(h_out):
                    for j in range(w_out):
                        im2col = pad_matrix[:, :, i:i + h_f, j:j + w_f].reshape(shape)
                        im2col_matrix.append(im2col)

                assert im2col_matrix

                im2col_matrix = Dense.concatenate(im2col_matrix, 0)
                shape = [batch_size * h_out * w_out, c_i * h_f * w_f]
                self.im2col_matrix = im2col_matrix.reshape(shape).transpose()
                self.w_col = self.subject.reshape([out_c, c_i * h_f * w_f])

            def __ns__(self, context, name_hint):
                Dual.__ns__(self, context, name_hint)
                deanonymize(self.im2col_matrix, context, name_hint + "_im2col_matrix")
                deanonymize(self.w_col, context, name_hint + "_w_col")

            def __repr__(self):
                return f"Convolution({self.subject}, {self.args})"

            def forward(self):
                return einsum("ij,jk->ik", [self.w_col, self.im2col_matrix])

            def backward(self, variable=None):
                w_col = derivative_of(self.w_col, variable, keepdims=True)
                im2col_matrix = derivative_of(self.im2col_matrix, variable, keepdims=True)
                return (w_col @ self.im2col_matrix) + (self.w_col @ im2col_matrix)

        shape = [out_c, h_out, w_out, batch_size]
        output = Tensor(Convolution(self.weights, inputs)) + self.bias
        output = output.reshape(shape).transpose([3, 0, 1, 2])  # shape = [batch_size, out_c, h_out, w_out]

        if self._activation:
            return self._activation(output)
        else:
            return output


class Linear(Layer, Dynamic):
    @classmethod
    def create(cls, input_size, output_size, activation=None, optimal_std=None):
        if callable(optimal_std):
            std = optimal_std(input_size, output_size)
        elif optimal_std:
            std = optimal_std
        else:
            std = (input_size * output_size)**0.5

        weights = Variable.random_normal([input_size, output_size], std=std)
        bias = Variable.random_normal([output_size], std=std)
        return cls(weights, bias, activation)

    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias
        self._activation = activation

        Dynamic.__init__(self)

    @differentiable
    def eval(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.shape[0]

        # TODO: this expectation should be set using a generic type
        inputs = Tensor.expect([batch_size, self.weights.shape[0]], Float)(form=inputs)

        x = (inputs @ self.weights) + self.bias
        if self._activation is None:
            return x
        else:
            return self._activation(x)


class NeuralNet(Model, Differentiable):
    """A neural network"""

    __uri__ = LIB_URI + "/NeuralNet"


class Sequential(NeuralNet, Dynamic):
    def __init__(self, layers):
        if not layers:
            raise ValueError("Sequential requires at least one layer")

        self.layers = layers
        Dynamic.__init__(self)

    @differentiable
    def eval(self, inputs: Tensor) -> Tensor:
        state = self.layers[0].eval(inputs)
        for i in range(1, len(self.layers)):
            state = self.layers[i].eval(state)

        return state


class DNN(Sequential):
    @classmethod
    def create(cls, schema):
        """
        Create a new :class:`Sequential` neural net of :class:`Linear` layers.

        `schema` should be a list of 2- or 3-tuples of the form `(input_size, output_size, activation)`
        (the arguments to `Linear.create`).
        """

        layers = Tuple([Linear.create(*layer_schema) for layer_schema in schema])
        return cls(layers)
