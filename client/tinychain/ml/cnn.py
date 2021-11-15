import math

from tinychain.collection.tensor import einsum, Dense
from tinychain.ml import Layer, Sigmoid
from tinychain.ref import After


class Conv(Layer):
    @property
    def bias(self):
        return Dense(self[1])

    @property
    def filters(self):
        return Dense(self[0])


def conv_1d(input_shape, filters, bias=None, stride=1, activation=Sigmoid()):
    if len(input_shape) != 2:
        raise ValueError(
            f"1D convolution requires an input shape of the form (length, input_channels), not {input_shape}")

    length, input_channels = input_shape
    padding = stride - 1

    class Conv1D(Conv):
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

            filtered = activation.forward(einsum("kio,blsi->blo", [self.filters, strides]))
            if bias is None:
                return filtered
            else:
                return filtered + self.bias

    return Conv1D([filters, bias])


def conv_2d(input_shape, filters, bias=None, stride=1, activation=Sigmoid()):
    if len(input_shape) != 3:
        raise ValueError(
            f"2D convolution requires an input shape of the form (width, height, input_channels), not {input_shape}")

    width, height, input_channels = input_shape
    padding = stride - 1

    class Conv2D(Conv):
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

            filtered = activation.forward(einsum("kkio,bwhssi->bwho", [self.filters, strides]))
            if bias is None:
                return filtered
            else:
                return filtered + self.bias

    return Conv2D([filters, bias])
