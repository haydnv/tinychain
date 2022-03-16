from turtle import forward
from client.tinychain.math.operator import Operator
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
        self.c_i, self.h_i, self.w_i = inputs_shape
        self.out_c, self.h_f, self.w_f = filter_shape
        # self._inputs_shape = inputs_shape
        # self._filter_shape = filter_shape
        self._stride = stride
        self._padding = padding
        self._activation = activation

        # run-time state
        self.weights = weights
        self.bias = bias

        Dynamic.__init__(self)

    @hidden
    def operator(self, inputs):
        x = Convolution(self, inputs)
        if self._activation:
            return self._activation(x)
        return x


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


#TODO: Convalution for ConvLayer.operator()
class Convolution(Operator):
    def __init__(self, params, inputs):
        Operator.__init__(self, params.weights, params.bias)
        self.params = params
        self.inputs = inputs

    def forward(self):
        # c_i, h_i, w_i = self.params._inputs_shape
        # out_c, h_f, w_f = self.params._filter_shape
        self.b_i = self.inputs.shape[0]
        self.h_out = int((self.params.h_i - self.params.h_f + 2 * self.params._padding) / self.params._stride + 1)
        self.w_out = int((self.params.w_i - self.params.w_f + 2 * self.params._padding) / self.params._stride + 1)

        assert self.h_out
        assert self.w_out

        pad_matrix = Dense.zeros([self.b_i, self.params.c_i, self.params.h_i + self.params._padding * 2, self.params.w_i + self.params._padding * 2])
        im2col_matrix = []
        for i in range(self.h_out):
            for j in range(self.w_out):
                im2col_matrix.append(pad_matrix[:, :, i:i + self.params.h_f, j:j + self.params.w_f].reshape([self.params.c_i * self.params.h_f * self.params.w_f, self.b_i]))
        im2col_concat = Tensor(After(pad_matrix[:, :, self.params._padding:(self.params._padding + self.params.h_i), self.params._padding:(self.params._padding + self.params.w_i)].write(self.inputs.copy()), Dense.concatenate(im2col_matrix, 0)))
        self.im2col_matrix = Tensor(After(im2col_concat, im2col_concat.reshape([self.b_i * self.params.h_out * self.params.w_out, self.params.c_i * self.params.h_f * self.params.w_f]).transpose()))
        w_col = self.subject.reshape([self.params.out_c, self.params.c_i * self.params.h_f * self.params.w_f])
        in2col_multiply = (w_col @ self.im2col_matrix + self.arg).reshape([self.params.out_c, self.params.h_out, self.params.w_out, self.b_i])
        output = Tensor(in2col_multiply.copy().transpose([3, 0, 1, 2]))

        if self._activation:
            return self._activation(output)

        return output

    def backward(self, dl):
        # delta = Tensor(self._activation.backward(Tensor(inputs)) * loss)
        delta_reshaped = Tensor(dl.transpose([1, 2, 3, 0])).reshape([self.params.out_c, self.params.h_out * self.params.w_out * self.b_i])
        self.dw = Tensor(einsum('ij,mj->im', [delta_reshaped, self.im2col_matrix])).reshape(self.subject.shape)
        self.db = Tensor(einsum('ijkb->j', [dl])).reshape([self.params.out_c, 1])
        dloss_col = Tensor(einsum('ji,jm->im', [self.subject.reshape([self.params.out_c, self.params.c_i * self.params.h_f * self.params.w_f]), delta_reshaped]))
        dloss_col_reshaped = dloss_col.reshape([self.params.c_i, self.params.h_f, self.params.w_f, self.params.h_out, self.params.w_out, self.b_i]).copy().transpose([5, 0, 3, 4, 1, 2])

        # TODO: make this a property of the ConvLayer instance
        pad_matrix = Dense.zeros([self.b_i, self.params.c_i, self.params.h_i + self.params._padding * 2, self.params.w_i + self.params._padding * 2])
        result = [
                pad_matrix[:, :, i:i + self.params.h_f, j:j + self.params.w_f].write(pad_matrix[:, :, i:i + self.params.h_f, j:j + self.params.w_f].copy() + dloss_col_reshaped[:, :, i, j, :, :])
                for i in range(self.h_out) for j in range(self.w_out)
                ]
        dloss_result = Tensor(After(result, pad_matrix[:, :, self.params._padding:(self.params._padding + self.params.h_i), self.params._padding:(self.params.padding + self.params.w_i)]))

        return dloss_result
