import math

from tinychain.collection.tensor import einsum, Dense
from tinychain.ml import Layer, Sigmoid
from tinychain.ref import After
from tinychain.state import Map


class Conv(Layer):
    ERR_ABSTRACT = "this is an abstract base class; use Conv1D or Conv2D instead"

    @classmethod
    def create(cls, num_spatial_dims, input_shape, output_shape, kernel_shape, stride=1, activation=Sigmoid()):
        input_channels = input_shape[-1]
        output_channels = output_shape[-1]

        weight_shape = kernel_shape * num_spatial_dims
        weight_shape += (input_channels, output_channels)

        bias = Dense.create([output_channels])
        weights = Dense.create(weight_shape)

        return cls.load(input_shape, weights, bias, stride, activation)

    @classmethod
    def load(cls, input_shape, weights, bias=0, stride=1, activation=Sigmoid()):
        raise NotImplementedError(cls.ERR_ABSTRACT)

    def eval(self):
        raise NotImplementedError(self.ERR_ABSTRACT)


class Conv1D(Conv):
    @classmethod
    def load(cls, input_shape, filters, bias=0, stride=1, activation=Sigmoid()):
        if len(input_shape) != 2:
            raise ValueError(
                f"1D convolution requires an input shape of the form (length, input_channels), not {input_shape}")

        length, input_channels = input_shape
        padding = stride - 1

        class _Conv1D(cls):
            def eval(self, inputs):
                padded = Dense.zeros([inputs.shape[0], length + padding, input_channels])
                l, r = padding // 2, padding // 2 if length % 2 == 0 else math.ceil(padding / 2)

                strides = []
                for x in range(length):
                    strides.append(padded[:, x:(x + stride)].expand_dims(1))

                strides = After(padded[:, l:-r].write(inputs), Dense.concatenate(strides, 1))

                # einsum dimensions:
                #   b = batch dim
                #   l = length
                #   i = input channels
                #   o = output channels
                #   k = kernel
                #   s = stride

                filtered = activation.forward(einsum("kio,blsi->blo", [self["filters"], strides]))
                return filtered + self["bias"]

        return _Conv1D(Map(filters=filters, bias=bias))

    def eval(self, inputs):
        raise NotImplementedError("use Conv1D.create or Conv1D.load to construct a 1D convolutional layer")


class Conv2D(Conv):
    @classmethod
    def load(cls, input_shape, filters, bias=0, stride=1, activation=Sigmoid()):
        if len(input_shape) != 3:
            raise ValueError(f"2D convolution requires an input shape of the form"
                             + " (width, height, input_channels), not {input_shape}")

        width, height, input_channels = input_shape
        padding = stride - 1

        class _Conv2D(cls):
            def eval(self, inputs):
                padded = Dense.zeros([inputs.shape[0], width + padding, height + padding, input_channels])

                l, r = padding // 2, padding // 2 if width % 2 == 0 else math.ceil(padding / 2)
                t, b = padding // 2, padding // 2 if height % 2 == 0 else math.ceil(padding / 2)

                strides = []
                for x in range(width):
                    x_strides = []
                    for y in range(height):
                        x_strides.append(padded[:, x:(x + stride), y:(y + stride)].expand_dims(1))

                    strides.append(Dense.concatenate(x_strides, 1).expand_dims(1))

                strides = After(padded[:, l:-r, t:-b].write(inputs), Dense.concatenate(strides, 1))

                # einsum dimensions:
                #   b = batch dim
                #   w = width
                #   h = height
                #   i = input channels
                #   o = output channels
                #   k = kernel
                #   s = stride

                filtered = activation.forward(einsum("kkio,bwhssi->bwho", [self["filters"], strides]))
                return filtered + self["bias"]

        return _Conv2D(Map(filters=filters, bias=bias))
