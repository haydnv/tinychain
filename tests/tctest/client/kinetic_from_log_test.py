
import tinychain as tc
from tinychain.util import form_of, hex_id
import torch
from torch.autograd import grad
import numpy as np

torch.manual_seed(11)
np.random.seed(11)


def kinetic_from_log(f, x):
    df_dy = form_of(f).gradients(tc.tensor.Dense.ones(f.shape))[hex_id(x)]
    lapl_elem = []

    def step(i: tc.UInt, _df_dy: tc.tensor.Tensor, x: tc.tensor.Tensor):
        _df_dy = df_dy[:, i]
        _lapl_elem = form_of(_df_dy).gradients(tc.tensor.Dense.ones(_df_dy.shape))[hex_id(x)][:, i]
        lapl_elem.append(_lapl_elem) 
    
    laplasian_list = tc.Stream.range(x.shape[1]).for_each(step)
    lapl_elem_matrix = tc.tensor.Dense(tc.After(laplasian_list, tc.tensor.Dense.concatenate(lapl_elem, 0)))

    lapl = lapl_elem_matrix.sum(axis=0) + (df_dy**2).sum(axis=-1)
    
    return form_of(df_dy).gradients(tc.tensor.Dense.ones(df_dy.shape))[hex_id(x)]#[:, 0]#tc.tensor.Tensor(lapl.expand_dims(-1) * (-0.5))


x = np.random.uniform(-1.0, 1.0, (6, 3))

def _hydrogen_torch(xs):
    return (-torch.norm(xs, dim=1, keepdim=True)).exp()

def _hydrogen_tensor(xs):
    return (-xs.norm(axis=1)).exp()

# def _helium(xs):
#     x1, x2 = torch.split(xs, 2, dim=1)
#     x1n = torch.norm(x1, dim=1, keepdim=True)
#     x2n = torch.norm(x2, dim=1, keepdim=True)
#     s = x1n + x2n
#     t = x1n - x2n
#     u = torch.norm(x1 - x2, dim=1, keepdim=True)
#     return torch.exp(s*-2) * (1 + 0.5*u*torch.exp(-1.013*u)) * (1 + 0.2119*s*u + 0.1406*t**2 - 0.003*u**2)


def kinetic_torch(f, x):
    df1 = grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    print(df1)
    lapl_elem = torch.cat([
        grad(df1[..., i].unsqueeze(-1),
            x,
            grad_outputs=torch.ones_like(df1[..., i]).unsqueeze(-1),
            create_graph=True,
            retain_graph=True
            )[0][..., i].unsqueeze(0)
        for i in range(df1.size()[1])
        ], 0)
    lapl = lapl_elem.sum(0) + (df1**2).sum(-1)
    return -0.5*lapl.unsqueeze(-1)


def test_hydrogen_atom_torch(x):
    xs_torch = torch.tensor(x).requires_grad_(True)
    logpsi_torch = torch.log(torch.abs(_hydrogen_torch(xs_torch)))
    kinetic_energy_torch = kinetic_torch(logpsi_torch, xs_torch)
    return kinetic_energy_torch

def test_hydrogen_atom_tensor(x):
    cxt = tc.Context()
    xs_tensor = tc.ml.optimizer.Variable.load(x.shape, x.flatten().tolist(), tc.F32)
    logpsi_tensor = _hydrogen_tensor(xs_tensor).abs().log()
    cxt.kinetic_energy_tensor = kinetic_from_log(logpsi_tensor, xs_tensor)
    print(HOST.post(ENDPOINT, cxt))
    #return load_np(HOST.post(ENDPOINT, cxt)) #kinetic_energy_tensor

def load_np(as_json, dtype=float):
    shape = as_json[TENSOR_URI][0][0]
    ndarray = np.array(as_json[TENSOR_URI][1], dtype)
    return ndarray.reshape(shape)


if __name__ == "__main__":
    TENSOR_URI = str(tc.uri(tc.tensor.Dense))
    HOST = tc.host.Host('http://127.0.0.1:8702')
    ENDPOINT = '/transact/hypothetical'

    print(test_hydrogen_atom_torch(x))
    print(test_hydrogen_atom_tensor(x))