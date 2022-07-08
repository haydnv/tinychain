""":class:`NeuralNet` and :class:`Layer` :class:`Model` definitions with common implementations"""

import inspect
import logging

from ..app import Dynamic, Model
from ..collection.tensor import Dense, Tensor
from ..context import Context
from ..decorators import differentiable, reflect
from ..math.operator import gradients
from ..generic import Map, Tuple
from ..reflect import method
from ..reflect.functions import parse_args
from ..scalar.number import Float
from ..scalar.ref import deref, form_of, is_ref, After, Post

from .constants import LIB_URI
from .interface import Differentiable, Gradients
from .variable import namespace, Variable


# TODO: move into the reflect.method module and rename
class ReflectedMethod(method.Post):
    def __init__(self, header, name, graph, sig, rtype):
        self.name = name
        self.header = header
        self.graph = graph
        self.sig = sig
        self.rtype = rtype

    def __call__(self, *args, **kwargs):
        params = parse_args(self.sig[1:], *args, **kwargs)
        return self.rtype(form=Post(self.subject(), params))

    def __form__(self):
        return self.graph


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
        self._activation = activation  # TODO: require a differentiable Function, not a callable Python literal

        # run-time state
        self.weights = weights
        self.bias = bias

        Dynamic.__init__(self)

    @differentiable
    def eval(self, cxt, inputs: Tensor) -> Tensor:
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

        pad_matrix = Dense.zeros([batch_size, c_i, h_i + padding * 2, w_i + padding * 2])
        pad_matrix = Tensor(After(
            pad_matrix[:, :, padding:(padding + h_i), padding:(padding + w_i)].write(inputs),
            pad_matrix))

        shape = [c_i * h_f * w_f, batch_size]

        im2col_matrix = []
        for i in range(h_out):
            for j in range(w_out):
                im2col = pad_matrix[:, :, i:i + h_f, j:j + w_f].reshape(shape)
                im2col_matrix.append(im2col)

        assert im2col_matrix

        shape = [batch_size * h_out * w_out, c_i * h_f * w_f]
        cxt.im2col_matrix = Dense.concatenate(im2col_matrix, 0).reshape(shape).transpose()
        cxt.w_col = self.weights.reshape([out_c, c_i * h_f * w_f])

        cxt.activation = (cxt.w_col @ cxt.im2col_matrix) + self.bias
        cxt.output = cxt.activation.reshape([out_c, h_out, w_out, batch_size]).transpose([3, 0, 1, 2])
        # shape is now [batch_size, out_c, h_out, w_out]

        return self._activation(cxt.output) if self._activation else cxt.output


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
        # compile-time constants
        self._activation = activation  # TODO: require a differentiable Function, not a callable Python literal

        # run-time state
        self.weights = weights
        self.bias = bias

        Dynamic.__init__(self)

    @differentiable
    def eval(self, cxt, inputs: Tensor) -> Tensor:
        batch_size = inputs.shape[0]

        # TODO: this expectation should be set using a generic type
        inputs = Tensor.expect([batch_size, self.weights.shape[0]], Float)(form=inputs)

        cxt.activation = inputs @ self.weights
        cxt.with_bias = cxt.activation + self.bias
        return self._activation(cxt.with_bias) if self._activation else cxt.with_bias

    @reflect
    def gradient(self, inputs: Tensor, loss: Tensor) -> Gradients:
        sig = list(inspect.signature(Linear.gradient).parameters.items())

        if is_ref(self.eval):
            form = deref(self.eval)
        else:
            form = form_of(self.eval)

        var_names = {var: name for name, var in namespace(self).items()}

        cxt = Context(form)
        cxt.grads = {var_names[var]: grad for var, grad in gradients(form[-1], loss).items() if var in var_names}
        return ReflectedMethod(self, "gradient", cxt, sig, Map.expect(Gradients))


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
