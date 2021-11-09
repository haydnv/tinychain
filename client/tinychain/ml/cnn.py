import math

from tinychain.collection.tensor import einsum, Dense
from tinychain.ml import Layer, Sigmoid
from tinychain.ref import After


def conv_1d(*args, **kwargs):
    raise NotImplementedError(f"1D convolution ({args}, {kwargs})")


def conv_2d(input_shape, filters, bias=None, stride=1, activation=Sigmoid()):
    if len(input_shape) != 3:
        raise ValueError(
            f"2D convolution requires an input shape of the form (width, height, input_channels), not {input_shape}")

    width, height, input_channels = input_shape
    padding = (stride - 1)

    class Conv2D(Layer):
        @property
        def bias(self):
            return Dense(self[1])

        @property
        def filters(self):
            return Dense(self[0])

        def eval(self, inputs):
            padded = Dense.zeros([inputs.shape()[0], width + padding, height + padding, input_channels])

            l, r = padding // 2, padding // 2 if width % 2 == 0 else math.ceil(padding / 2)
            t, b = padding // 2, padding // 2 if height % 2 == 0 else math.ceil(padding / 2)

            strides = []
            for x in range(width):
                x_strides = []
                for y in range(height):
                    x_strides.append(padded[:, x:(x + stride), y:(y + stride)].expand_dims(1))

                strides.append(Dense.concatenate(x_strides, 1).expand_dims(1))

            strides = After(
                padded.write([slice(None), slice(l, -r), slice(t, -b)], inputs),
                Dense.concatenate(strides, 1))

            # einsum dimensions:
            #   b = batch dim
            #   w = width
            #   h = height
            #   i = input channels
            #   o = output channels
            #   k = kernel
            #   s = stride

            if bias is None:
                return activation.forward(einsum("kkio,bwhssi->bwho", [self.filters, strides]))
            else:
                return activation.forward(einsum("kkio,bwhssi->bwho", [self.filters, strides])) + self.bias

    return Conv2D([filters, bias])
