from typing import List, Tuple

from tinychain.collection.tensor import einsum, Dense, Tensor
from tinychain.ml import Layer, Sigmoid, DiffedParameter, Parameter
from tinychain.ref import After
from client.tinychain.value import Int

#TODO: Implementation ConvLayer.forward and ConvLayer.backward
class ConvLayer(Layer):
    # @classmethod
    # def create(cls, name: str, inputs_shape, filter_shape, stride=1, padding=1, activation=Sigmoid()):
    #     """
    #     `inputs_stape`: size of inputs [b_i, c_i, h_i, w_i] where
    #         `b_i`: batch_size;
    #         `c_i`: number of channels;
    #         `h_i`: height's width  for each channel ;
    #         'w_i': matrix's width  for each channel .
    #     `filter_shape`: size of filter [h, w, out_c] where
    #         `h_f`: height of the kernel
    #         'w_f`: width of the kernel
    #         `out_c`: number of output channels
    #     """
    #     b_i, c_i, h_i, w_i = inputs_shape
    #     h_f, w_f, out_c = filter_shape
    #     weight = Dense.create([out_c, c_i, h_f, w_f])
    #     bias = Dense.create([out_c])

    #     return cls.load(name, weight, bias, stride, padding, activation)

    @classmethod
    def load(cls, name, weights, bias, stride=1, padding=1, activation=Sigmoid()):
        """Load a `DNNLayer` with the given `weights` and `bias` tensors."""
        class _ConvLayer(cls):

            def forward(self, x: Dense):
                out_c, c_i, h_f, w_f = weights.shape.unpack(4)
                b_i, c_i, h_i, w_i = x.shape.unpack(4)
                h_out = int((3 - 3 + 2 * padding) / stride + 1)
                w_out = int((3 - 3 + 2 * padding) / stride + 1)
                pad_matrix = Dense.zeros([b_i, c_i, 3 + padding * 2, 3 + padding * 2])
                im2col_matrix = []
                for i in range(h_out):
                    for j in range(w_out):
                        im2col_matrix.append(pad_matrix[:, :, i:i+3, j:j+3].reshape([3*3*2, 1]))
                im2col_matrix = Tensor(After(pad_matrix[:, :, padding:-padding, padding:-padding].write(x.copy()), Dense.concatenate(col2im_list, 1)))
                w_col = self[name+'.weights'].reshape([Int(out_c), Int(h_f)*Int(w_f)*Int(c_i)])
                in2col_multiply = Tensor(einsum("ij,jm->im", [w_col, im2col_matrix]) + self[name+'.bias']).reshape([out_c, h_out, w_out, b_i])
                output = Tensor(in2col_multiply.copy().transpose([3, 0, 1, 2]))

                return activation.forward(output), im2col_matrix

            def backward(self, x, loss):
                inputs_data, im2col_matrix = x.unpack(2)
                out_c = self[name+'.weights'].shape[0]
                inputs, _ = self.forward(inputs_data)
                delta = Tensor(activation.backward(inputs) * loss)
                delta_reshaped = delta.transpose(1, 2, 3, 0).reshape([out_c, 1])
                dw = Tensor(einsum('ij,mj->im', [delta_reshaped, im2col_matrix])).reshape(self[name+'.weights'].shape)
                db = Tensor(delta.sum([0, 2, 3])).reshaped([out_c, 1])
                dloss_col = Tensor(einsum('ji,jm->im', [self[name+'.weights'].reshape([out_c, 1]), delta_reshaped])).reshape([inputs_data.shape]+[3, 3])
                ### write col2im
                h_out = int((3 - 3 + 2 * padding) / stride + 1)
                w_out = int((3 - 3 + 2 * padding) / stride + 1)
                pad_matrix = Dense.zeros([b_i, c_i, 3 + padding * 2, 3 + padding * 2])
                im2col_matrix = []
                for i in range(h_out):
                    for j in range(w_out):
                        pad_matrix[:, :, i:i+3, j:j+3].write(pad_matrix[:, :, i:i+3, j:j+3] + dloss_col[:, :, i:i+3, j:j+3])
                ###
                return pad_matrix[:, :, padding:-padding, padding:-padding], [
                    DiffedParameter.create(
                        name=name + '.weight',
                        value=self[name + ".weights"],
                        grad=dw),
                    DiffedParameter.create(
                        name=name + '.bias',
                        value=self[name + ".bias"],
                        grad=db)
                ]

            def get_param_list(self) -> List[Parameter]: 
                return [
                    Parameter.create(name=name + '.weight', value=self[name + ".weights"]),
                    Parameter.create(name=name + '.bias', value=self[name + ".bias"])
                ]

        return _ConvLayer({name + ".weights": weights, name + ".bias": bias})
