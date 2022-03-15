from ..app import Dynamic, Model
from ..collection.tensor import einsum, Dense, Tensor
from ..decorators import hidden, post
from ..generic import Tuple
from ..ml import Activation
from ..scalar.ref import After

from .interface import Differentiable
from .optimizer import Variable
from . import LIB_URI


class Layer(Model, Differentiable):
    """A :class:`Layer` in a :class:`NeuralNet`"""

    __uri__ = LIB_URI + "/Layer"


class ConvLayer(Layer, Dynamic):
    @classmethod
    def create(cls, inputs_shape, filter_shape, stride=1, padding=1, activation=None):
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

        optimal_std = activation.optimal_std if activation else Activation.optimal_std
        std = optimal_std(c_i * h_i * w_i, out_c * h_f * w_f)
        weights = Variable.random_normal([out_c, c_i, h_f, w_f], mean=0.0, std=std)
        bias = Variable.random_normal([out_c, 1], mean=0.0, std=std)

        return cls(weights, bias, inputs_shape, filter_shape, stride, padding, activation)

    def __init__(self, weights, bias, inputs_shape, filter_shape, stride, padding, activation):
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
        c_i, h_i, w_i = self._inputs_shape
        out_c, h_f, w_f = self._filter_shape
        b_i = inputs.shape[0]
        h_out = int((h_i - h_f + 2 * self._padding) / self._stride + 1)
        w_out = int((w_i - w_f + 2 * self._padding) / self._stride + 1)

        assert h_out
        assert w_out

        pad_matrix = Dense.zeros([b_i, c_i, h_i + self._padding * 2, w_i + self._padding * 2])
        im2col_matrix = []
        for i in range(h_out):
            for j in range(w_out):
                im2col_matrix.append(pad_matrix[:, :, i:i + h_f, j:j + w_f].reshape([c_i * h_f * w_f, b_i]))
        im2col_concat = Tensor(After(pad_matrix[:, :, self._padding:(self._padding + h_i), self._padding:(self._padding + w_i)].write(inputs.copy()), Dense.concatenate(im2col_matrix, 0)))
        im2col_matrix = Tensor(After(im2col_concat, im2col_concat.reshape([b_i * h_out * w_out, c_i * h_f * w_f]).transpose()))
        w_col = self.weights.reshape([out_c, c_i * h_f * w_f])
        in2col_multiply = (w_col @ im2col_matrix + self.bias).reshape([out_c, h_out, w_out, b_i])
        output = Tensor(in2col_multiply.copy().transpose([3, 0, 1, 2]))

        if self._activation:
            return self._activation(output)

        return output

    # @post
    # def eval(self, cxt, inputs: Tensor) -> Tensor:
    #     b_i = inputs.shape[0]

    #     if self._padding == 0:
    #         output = einsum("abcd,efgh->eacd", [self.weights, inputs])
    #         output += self.bias.reshape([1, self._filter_shape[0], 1, 1])
    #         if self._activation:
    #             return self._activation(output).forward()
    #         else:
    #             return output

    #     padding = self._padding
    #     stride = self._stride

    #     c_i, h_i, w_i = self._inputs_shape
    #     out_c, h_f, w_f = self._filter_shape

    #     h_out = int(((h_i - h_f) + (2 * padding)) / (stride + 1))
    #     w_out = int(((w_i - w_f) + (2 * padding)) / (stride + 1))

    #     assert h_out
    #     assert w_out

    #     _pad_matrix = Dense.zeros([b_i, c_i, h_i + padding * 2, w_i + padding * 2])
    #     cxt.pad_matrix = Tensor(After(
    #         _pad_matrix[:, :, padding:(padding + h_i), padding:(padding + w_i)].write(inputs),
    #         _pad_matrix))

    #     _im2col_matrix = []
    #     for i in range(h_out):
    #         for j in range(w_out):
    #             shape = [c_i * h_f * w_f, b_i]
    #             _im2col = cxt.pad_matrix[:, :, i:i + h_f, j:j + w_f].reshape(shape)
    #             _im2col_matrix.append(_im2col)

    #     assert _im2col_matrix

    #     shape = [b_i * h_out * w_out, c_i * h_f * w_f]
    #     cxt.im2col_matrix = Dense.concatenate(_im2col_matrix, 0).reshape(shape).transpose()
    #     cxt.w_col = self.weights.reshape([out_c, c_i * h_f * w_f])

    #     shape = [out_c, h_out, w_out, b_i]
    #     cxt.in2col_multiply = (einsum("ij,jm->im", [cxt.w_col, cxt.im2col_matrix]) + self.bias).reshape(shape)

    #     output = cxt.in2col_multiply.copy().transpose([3, 0, 1, 2])  # shape = [b_i, out_c, h_out, w_out]

    #     if self._activation:
    #         return self._activation(output).forward()
    #     else:
    #         return output


class DNNLayer(Layer, Dynamic):
    @classmethod
    def create(cls, input_size, output_size, activation=None):
        optimal_std = activation.optimal_std if activation else Activation.optimal_std
        std = optimal_std(input_size, output_size)
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
