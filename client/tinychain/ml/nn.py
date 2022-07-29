""":class:`NeuralNet` and :class:`Layer` :class:`Model` definitions with common implementations"""

import inspect
import logging
import typing

from ..app import Dynamic, Model
from ..collection.tensor import einsum, Dense, Tensor
from ..context import Context
from ..decorators import differentiable, post, reflect
from ..error import NotImplemented
from ..math.operator import derivative_of, gradients, Dual, Gradients
from ..generic import Map, Tuple
from ..reflect import method
from ..reflect.functions import parse_args
from ..scalar.number import Float
from ..scalar.ref import deref, form_of, get_ref, is_ref, After
from ..uri import URI

from .constants import LIB_URI
from .interface import Differentiable, Gradient
from .variable import namespace, Variable


# TODO: move into the reflect.method module and rename
class ReflectedMethod(method.Post):
    def __init__(self, header, name, graph, sig, rtype):
        if tuple(sig.parameters)[0] != "self":
            raise TypeError(f"not a method signature: {tuple(sig.parameters.items())}")

        self.name = name
        self.header = header
        self.graph = graph
        self.sig = sig
        self.rtype = rtype

    def __call__(self, *args, **kwargs):
        from ..scalar.ref import Post
        params = parse_args(self.sig[1:], *args, **kwargs)
        return self.rtype(form=Post(self.subject(), params))

    def __form__(self):
        return self.graph


class Layer(Model, Differentiable):
    """A :class:`Layer` in a :class:`NeuralNet`"""

    __uri__ = LIB_URI + "/Layer"

    @reflect
    def gradient(self, inputs: Tensor, loss: Tensor) -> Map[Gradient]:
        if self.eval is Layer.eval:
            # if this is an abstract class, don't try to reflect over the eval method
            return NotImplemented("Layer.gradient")

        sig = inspect.signature(Linear.gradient.form)

        if is_ref(self.eval):
            form = deref(self.eval)
        else:
            form = form_of(self.eval)

        var_names = {var: name for name, var in namespace(self).items()}
        grads = gradients(form[-1], loss)
        [loss] = [grad for var, grad in grads.items() if var not in var_names]
        grads = {var_names[var]: grad for var, grad in grads.items() if var in var_names}
        grads["inputs"] = loss

        cxt = Context(form)
        cxt.gradient = grads

        return ReflectedMethod(self, "gradient", cxt, sig, Map[Gradient])


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

        inputs = Tensor[Float].with_shape([batch_size] + self._inputs_shape)(form=inputs)

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

        im2col_matrix = Dense.concatenate(im2col_matrix, 0)
        cxt.im2col_matrix = im2col_matrix.reshape([batch_size * h_out * w_out, c_i * h_f * w_f])
        cxt.im2col_matrix_T = cxt.im2col_matrix.transpose()

        cxt.w_col = self.weights.reshape([out_c, c_i * h_f * w_f])

        class Convolution(Dual):
            def __repr__(self):
                return f"Convolution({self.subject}, {self.args})"

            def forward(self):
                return einsum("ij,jk->ik", [cxt.w_col, cxt.im2col_matrix_T])

            def backward(self, variable=None):
                w_col = derivative_of(cxt.w_col, variable, keepdims=True)
                im2col_matrix = derivative_of(cxt.im2col_matrix_T, variable, keepdims=True)
                return (w_col @ cxt.im2col_matrix_T) + (cxt.w_col @ im2col_matrix)

            def gradients(self, loss):
                grads = Gradients()

                grads[self.subject] = (loss @ cxt.im2col_matrix).reshape(self.subject.shape)

                loss = (cxt.w_col.transpose() @ loss)
                # TODO: there should not be any loss of precision in this step
                loss = loss.reshape([batch_size, c_i, None]).sum(-1).expand_dims().expand_dims()
                grads[self.args] = Dense.zeros([batch_size, c_i, h_i, w_i])
                grad_slice = grads[self.args][:, :, padding:(h_i - padding), padding:(w_i - padding)]
                grads[self.args] = Tensor(After(grad_slice.write(loss), grads[self.args]))

                return grads

        shape = [out_c, h_out, w_out, batch_size]
        cxt.activation = Tensor(Convolution(self.weights, inputs)) + self.bias
        cxt.output = cxt.activation.reshape(shape).transpose([3, 0, 1, 2])  # shape = [batch_size, out_c, h_out, w_out]

        if self._activation:
            return self._activation(cxt.output)
        else:
            return cxt.output


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

        inputs = Tensor[Float].with_shape([batch_size, self.weights.shape[0]])(form=inputs)

        cxt.activation = inputs @ self.weights
        cxt.with_bias = cxt.activation + self.bias
        return self._activation(cxt.with_bias) if self._activation else cxt.with_bias


class NeuralNet(Model, Differentiable):
    """A neural network"""

    __uri__ = LIB_URI + "/NeuralNet"


class Sequential(NeuralNet, Dynamic):
    """A sequence of :class:`Layer` s"""

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

    @post
    def gradient(self, cxt, inputs: Tensor, loss: Tensor) -> Map[Gradient]:
        layer_inputs = [inputs]

        for layer in self.layers[:-1]:
            layer_inputs.append(layer.eval(layer_inputs[-1]))

        cxt.layer_inputs = layer_inputs

        layer_grads = []
        for i, (inputs, layer) in reversed(list(enumerate(zip(cxt.layer_inputs, self.layers)))):
            # TODO: this call to get_ref should not be necessary
            layer = get_ref(layer, URI(self, "layers", i))
            layer_grad = layer.gradient(inputs, loss)
            layer_grads.append(layer_grad)
            loss = layer_grad["inputs"]  # TODO: should this handle other layer eval signatures automatically?

        cxt.layer_grads = list(reversed(layer_grads))

        grads = {}
        for i, layer_grad in enumerate(cxt.layer_grads):
            for name in namespace(self.layers[i]):
                grads[f"layers.{i}.{name}"] = layer_grad[name]

        grads["inputs"] = loss

        return grads


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
