import typing

from ..app import Dynamic, Model
from ..collection.tensor import einsum, Dense, Tensor
from ..decorators import post_method
from ..ml import Activation
from ..state.generic import Tuple
from ..state.ref import After
from ..util import uri

from .interface import Differentiable
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

        optimal_std = activation.optimal_std if activation else Activation.optimal_std
        std = optimal_std(c_i * h_i * w_i, out_c * h_f * w_f)
        weights = Dense.random_normal([out_c, c_i, h_f, w_f], mean=0.0, std=std)
        bias = Dense.random_normal([out_c, 1], mean=0.0, std=std)

        return cls(weights, bias, inputs_shape, filter_shape, stride, padding, activation)

    def __init__(self, weights, bias, inputs_shape, filter_shape, stride, padding, activation):
        self.weights = weights
        self.bias = bias
        self.inputs_shape = inputs_shape
        self.filter_shape = filter_shape
        self.stride = stride
        self.padding = padding
        self._activation = activation

        Dynamic.__init__(self)

    @post_method
    def forward(self, cxt, inputs: Dense):
        padding = self.padding
        stride = self.stride

        c_i, h_i, w_i = self.inputs_shape
        out_c, h_f, w_f = self.filter_shape
        b_i = inputs.shape[0]

        h_out = int((h_i - h_f + 2 * padding) / (stride + 1))
        w_out = int((w_i - w_f + 2 * padding) / (stride + 1))

        _pad_matrix = Dense.zeros([b_i, c_i, h_i + padding * 2, w_i + padding * 2])
        cxt.pad_matrix = Tensor(After(_pad_matrix[:, :, padding:(padding + h_i), padding:(padding + w_i)].write(inputs), _pad_matrix))

        _im2col_matrix = []
        for i in range(h_out):
            for j in range(w_out):
                _im2col = cxt.pad_matrix[:, :, i:i + h_f, j:j + w_f].reshape([c_i * h_f * w_f, b_i])
                _im2col_matrix.append(_im2col)

        shape = [b_i * h_out * w_out, c_i * h_f * w_f]
        cxt.im2col_matrix = Dense.concatenate(_im2col_matrix, 0).reshape(shape).transpose()
        cxt.w_col = self.weights.reshape([out_c, c_i * h_f * w_f])

        shape = [out_c, h_out, w_out, b_i]
        cxt.in2col_multiply = (einsum("ij,jm->im", [cxt.w_col, cxt.im2col_matrix]) + self.bias).reshape(shape)

        output = cxt.in2col_multiply.copy().transpose([3, 0, 1, 2])

        if self._activation:
            return self._activation.forward(output)
        else:
            return output


class DNNLayer(Layer, Dynamic):
    @classmethod
    def create(cls, input_size, output_size, activation=None):
        optimal_std = activation.optimal_std if activation else Activation.optimal_std
        std = optimal_std(input_size, output_size)
        weights = Dense.random_normal([input_size, output_size], std=std)
        bias = Dense.random_normal([output_size], std=std)
        return cls(weights, bias, activation)

    def __init__(self, weights, bias, activation=None):
        self.weights = weights
        self.bias = bias
        self._activation = activation

        Dynamic.__init__(self)

    @post_method
    def forward(self, inputs: Tensor) -> Tensor:
        x = einsum("ki,ij->kj", [inputs, self.weights]) + self.bias
        if self._activation is None:
            return x
        else:
            return self._activation.forward(x)


class NeuralNet(Model, Differentiable):
    """A neural network"""

    __uri__ = LIB_URI + "/NeuralNet"


class Sequential(NeuralNet, Dynamic):
    def __init__(self, layers):
        if not layers:
            raise ValueError("Sequential requires at least one layer")

        self.layers = layers
        Dynamic.__init__(self)

    @post_method
    def forward(self, inputs: typing.Tuple[Tensor]) -> Tensor:
        if uri(self.layers[0]) != "$self/layers/0":
            raise RuntimeError(f"{self.__class__.__name__}.forward must be called with a header (URI {uri(self.layers[0])} is not valid)")

        state = self.layers[0].forward(inputs=inputs)
        for i in range(1, len(self.layers)):
            state = self.layers[i].forward(inputs=state)

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
