from typing import List

from tinychain.collection.tensor import einsum, Dense, Tensor
from tinychain.ml import Layer, Sigmoid, DiffedParameter, Parameter
from tinychain.ref import After


class ConvLayer(Layer):
    @classmethod
    def create(cls, name: str, inputs_shape, filter_shape, stride=1, padding=1, activation=Sigmoid()):
        """
        `inputs_stape`: size of inputs [b_i, c_i, h_i, w_i] where
            `b_i`: batch_size;
            `c_i`: number of channels;
            `h_i`: height's width  for each channel ;
            'w_i': matrix's width  for each channel .
        `filter_shape`: size of filter [h, w, out_c] where
            `h_f`: height of the kernel
            'w_f`: width of the kernel
            `out_c`: number of output channels
        """
        b_i, c_i, h_i, w_i = inputs_shape
        h_f, w_f, out_c = filter_shape
        weight = Dense.create([out_c, c_i, h_f, w_f])
        bias = Dense.create([out_c])
        
        return cls.load(name, weight, bias, stride, padding, activation)

    @classmethod
    def load(cls, name, weights, bias, stride, padding, activation=Sigmoid()):
        """Load a `DNNLayer` with the given `weights` and `bias` tensors."""

        class _ConvLayer(cls):

            def forward(self, x):
                out_c, c_i, h_f, w_f = self['weight'].shape
                b_i, c_i, h_i, w_i = x.shape

                h_out = (h_i - h_f + 2 * padding) / stride + 1
                w_out = (w_i - w_f + 2 * padding) / stride + 1

                pad_zeros = Dense.zeros([b_i, c_i, h_i + padding * 2, w_i + padding * 2])

                pad_matrix = After(pad_zeros[:, :, padding:-padding, padding:-padding].write(x), pad_zeros)

                col2im_list = []
                for i in range(h_i):
                    for j in range(w_f):
                        col2im_list.append(pad_matrix[:, :, i:i+h_f, j:j+w_f].reshape(-1))
                col2im_matrix = Dense.concatenate(col2im_list, 1)
                #col2im_matrix.shape[h_i*w_i, b_i*c_i*h_f*w_f]

                w_col = self['weight'].reshape(self['weight'].shape[0] + self['weight'].shape[1], -1)
                output = einsum("ij,mj->im", [w_col, col2im_matrix]) + self['bias']
                output = output.reshape(out_c, h_out, w_out, b_i)

                return Tensor(einsum('fhwb->bfhw', [output]))

            def backward():
                pass

            def get_param_list(self) -> List[Parameter]: 
                pass

        return _ConvLayer({name + ".weights": weights, name + ".bias": bias})
