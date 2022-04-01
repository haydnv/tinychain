import logging

from ..app import Dynamic, Model
from ..collection.tensor import einsum, Dense, NDArray, Tensor, Transform
from ..decorators import hidden
from ..generic import Tuple
from ..math import Operator
from ..scalar.ref import After
from ..util import deanonymize, form_of

from .interface import Differentiable
from .variable import Variable
from . import LIB_URI


class Layer(Model, Differentiable):
    """A :class:`Layer` in a :class:`NeuralNet`"""

    __uri__ = LIB_URI + "/Layer"


class ConvLayer(Layer, Dynamic):
    @classmethod
    def create(cls, inputs_shape, filter_shape, stride=1, padding=1, activation=None, optimal_std=None):
        """
        Create a new, empty `ConvLayer` with the given shape and activation function.

        Args:
            `inputs_shape`: size of inputs [c_i, h_i, w_i] where
                `c_i`: number of channels;
                `h_i`: height's width for each channel;
                'w_i': matrix's width for each channel.
            `filter_shape`: size of filter [h_f, w_f, out_c] where
                `out_c`: number of output channels;
                `h_f`: height of the kernel;
                'w_f`: width of the kernel.
            `activation`: activation function.
        """

        c_i, h_i, w_i = inputs_shape
        out_c, h_f, w_f = filter_shape

        input_size = c_i * h_i * w_i
        output_size = out_c * h_f * w_f
        std = optimal_std(input_size, output_size) if optimal_std else (input_size * output_size)**0.5
        weights = Variable.random_normal([out_c, c_i, h_f, w_f], mean=0.0, std=std)
        bias = Variable.random_normal([out_c, 1], mean=0.0, std=std)

        return cls(weights, bias, inputs_shape, filter_shape, stride, padding, activation)

    def __init__(self, weights, bias, inputs_shape, filter_shape, stride, padding, activation):
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

    @hidden
    def operator(self, inputs):
        b_i = inputs.shape[0]

        if self._padding == 0:
            output = einsum("abcd,efgh->eacd", [self.weights, inputs])  # TODO: define an Operator for this case
            output += self.bias.reshape([1, self._filter_shape[0], 1, 1])
            if self._activation:
                return self._activation(output)
            else:
                return output

        padding = self._padding
        stride = self._stride

        c_i, h_i, w_i = self._inputs_shape
        out_c, h_f, w_f = self._filter_shape

        h_out = int(((h_i - h_f) + (2 * padding)) / (stride + 1))
        w_out = int(((w_i - w_f) + (2 * padding)) / (stride + 1))

        assert h_out
        assert w_out

        class Convolution(Transform):
            def __init__(self, weights, inputs):
                Transform.__init__(self, weights, inputs)

                pad_matrix = Dense.zeros([b_i, c_i, h_i + padding * 2, w_i + padding * 2])
                pad_matrix = Tensor(After(
                    pad_matrix[:, :, padding:(padding + h_i), padding:(padding + w_i)].write(self.args),
                    pad_matrix))

                im2col_matrix = []
                for i in range(h_out):
                    for j in range(w_out):
                        shape = [c_i * h_f * w_f, b_i]
                        im2col = NDArray.reshape(pad_matrix[:, :, i:i + h_f, j:j + w_f], shape)
                        im2col_matrix.append(im2col)

                assert im2col_matrix

                shape = [b_i * h_out * w_out, c_i * h_f * w_f]
                im2col_matrix = Dense.concatenate(im2col_matrix, 0)

                self.im2col_matrix = im2col_matrix.reshape(shape).transpose()
                self.w_col = self.subject.reshape([out_c, c_i * h_f * w_f])

            def __ns__(self, context):
                deanonymize(self.im2col_matrix, context)
                deanonymize(self.w_col, context)

            def forward(self):
                return einsum("ij,jk->ik", [self.w_col, self.im2col_matrix])

            def gradients(self, loss):
                grads = self.w_col.invert(loss @ self.im2col_matrix.transpose())

                if isinstance(form_of(self.args), Operator):
                    loss = (self.w_col.transpose() * loss).reshape([b_i, c_i, h_f, w_f])
                    grad = Dense.zeros([b_i, c_i, h_i, w_i])
                    grad = Tensor(After(grad[:, :, padding:(h_i - padding), padding:(w_i - padding)].write(loss), grad))
                    grads.update(form_of(self.args).gradients(grad))

                return grads

        shape = [out_c, h_out, w_out, b_i]
        in2col_multiply = Tensor(Convolution(self.weights, inputs)) + self.bias
        output = in2col_multiply.reshape(shape).transpose([3, 0, 1, 2])  # shape = [b_i, out_c, h_out, w_out]

        if self._activation:
            return self._activation(output)
        else:
            return output


class DNNLayer(Layer, Dynamic):
    @classmethod
    def create(cls, input_size, output_size, activation=None, optimal_std=None):
        std = optimal_std(input_size, output_size) if optimal_std else (input_size * output_size)**0.5
        weights = Variable.random_normal([input_size, output_size], std=std)
        bias = Variable.random_normal([output_size], std=std)
        return cls(weights, bias, activation)

    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias
        self._activation = activation

        Dynamic.__init__(self)

    @hidden
    def operator(self, inputs):
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

    @hidden
    def operator(self, inputs):
        state = self.layers[0].operator(inputs)
        for i in range(1, len(self.layers)):
            state = self.layers[i].operator(state)

        return state


class DNN(Sequential):
    @classmethod
    def create(cls, schema):
        """
        Create a new :class:`Sequential` neural net of :class:`DNNLayer` s.

        `schema` should be a list of 2- or 3-tuples of the form `(input_size, output_size, activation)`
        (the arguments to `DNNLayer.create`).
        """

        layers = Tuple([DNNLayer.create(*layer_schema) for layer_schema in schema])
        return cls(layers)
