from typing import List

from tinychain.collection.tensor import einsum, Dense, Tensor
from tinychain.ml import Layer, Blank, Sigmoid, Tanh, ReLU, DiffedParameter, Parameter
from tinychain.ref import After

#TODO: Implementation ConvLayer.forward and ConvLayer.backward 
class ConvLayer(Layer):
    @classmethod
    def create(cls, name: str, inputs_shape, filter_shape, stride=1, padding=1, activation=Blank()):
        """
        `inputs_stape`: size of inputs [b_i, c_i, h_i, w_i] where
            `b_i`: batch_size;
            `c_i`: number of channels;
            `h_i`: height's width  for each channel ;
            'w_i': matrix's width  for each channel .
        `filter_shape`: size of filter [h, w, out_c] where
            `out_c`: number of output channels;
            `h_f`: height of the kernel;
            'w_f`: width of the kernel.
        """

        b_i, c_i, h_i, w_i = inputs_shape
        out_c, h_f, w_f  = filter_shape

        if activation == Blank():
            a = (out_c*c_i*h_f*w_f)**(-0.5)
            gain = 1
        elif activation == Sigmoid():
            a = (2/(c_i*h_i*w_i + out_c*h_f*w_f))**0.5
            gain = 1
        elif activation == Tanh():
            a = (2/(c_i*h_i*w_i + out_c*h_f*w_f))**0.5
            gain = 5/3
        elif activation == ReLU():
            a = ((c_i*h_i*w_i)/(out_c*h_f*w_f))**(-0.5)
            gain = 2**0.5
        std = gain * a
        weight = Dense.random_uniform(shape=[out_c, c_i, h_f, w_f], 0, std)
        bias = Dense.random_uniform(shape=[out_c , 1], 0, std)

        return cls.load(name, weight, bias, inputs_shape, filter_shape, stride, padding, activation)

    @classmethod
    def load(cls, name, weights, bias, inputs_shape, filter_shape, stride, padding, activation):
        """Load a `DNNLayer` with the given `weights` and `bias` tensors."""
        class _ConvLayer(cls):

            def forward(self, x: Dense):
                b_i, c_i, h_i, w_i = inputs_shape#4, 3, 3, 3
                out_c, h_f, w_f = filter_shape#2, 3, 5, 5
                h_out = int((h_i - h_f + 2 * padding) / stride + 1)
                w_out = int((w_i - w_f + 2 * padding) / stride + 1)
                pad_matrix = Dense.zeros([b_i, c_i, h_i + padding * 2, w_i + padding * 2])
                im2col_matrix = []
                for i in range(h_out):
                    for j in range(w_out):
                        im2col_matrix.append(pad_matrix[:, :, i:i+h_f, j:j+w_f].reshape([c_i*h_f*w_f, b_i]))
                im2col_concat = Tensor(After(pad_matrix[:, :, padding:-padding, padding:-padding].write(x.copy()), Dense.concatenate(im2col_matrix, 0)))
                im2col_matrix = Tensor(After(im2col_concat, im2col_concat.reshape([b_i*h_i*w_i, c_i*h_f*w_f]).transpose()))
                w_col = self[name+'.weights'].reshape([out_c, c_i*h_f*w_f])
                in2col_multiply = Tensor(einsum("ij,jm->im", [w_col, im2col_matrix]) + self[name+'.bias']).reshape([out_c, h_out, w_out, b_i])
                output = Tensor(in2col_multiply.copy().transpose([3, 0, 1, 2]))

                return activation.forward(output), im2col_matrix

            def backward(self, x, loss):
                b_i, c_i, h_i, w_i = inputs_shape#4, 3, 3, 3
                out_c, h_f, w_f = filter_shape#2, 3, 5, 5
                h_out = int((h_i - h_f + 2 * padding) / stride + 1)
                w_out = int((w_i - w_f + 2 * padding) / stride + 1)
                inputs, im2col_matrix = self.forward(x)
                delta = Tensor(activation.backward(Tensor(inputs)) * loss)
                delta_reshaped = delta.transpose([1, 2, 3, 0]).reshape([out_c, h_i*w_i*b_i])
                dw = Tensor(einsum('ij,mj->im', [delta_reshaped, im2col_matrix])).reshape(self[name+'.weights'].shape)
                db = Tensor(einsum('ijkb->j', [delta])).reshape([out_c, 1])
                dloss_col = Tensor(einsum('ji,jm->im', [self[name+'.weights'].reshape([out_c, c_i*h_f*w_f]), delta_reshaped]))
                dloss_col_reshaped = dloss_col.reshape([c_i, h_f, w_f, h_out, w_out, b_i]).copy().transpose([5, 0, 3, 4, 1, 2])
                pad_matrix = Dense.zeros([b_i, c_i, h_i + padding * 2, w_i + padding * 2])
                result = [pad_matrix[:, :, i:i+h_f, j:j+w_f].write(pad_matrix[:, :, i:i+h_f, j:j+w_f].copy() + dloss_col_reshaped[:, :, i, j, :, :])
                for i in range(h_out) for j in range(w_out)]
                dloss_result = After(result, pad_matrix[:, :, padding:-padding, padding:-padding])

                return dloss_result, [
                    DiffedParameter.create(
                        name=name + '.weights',
                        value=self[name + ".weights"],
                        grad=dw),
                    DiffedParameter.create(
                        name=name + '.bias',
                        value=self[name + ".bias"],
                        grad=db)
                ]

            def get_param_list(self) -> List[Parameter]: 
                return [
                    Parameter.create(name=name + '.weights', value=self[name + ".weights"]),
                    Parameter.create(name=name + '.bias', value=self[name + ".bias"])
                ]

        return _ConvLayer({name + ".weights": weights, name + ".bias": bias})
