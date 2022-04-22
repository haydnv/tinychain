import numpy as np
import unittest
import testutils
import tinychain as tc
import torch
from torch.autograd import grad
from tinychain.util import form_of, hex_id

TENSOR_URI = str(tc.uri(tc.tensor.Dense))
HOST = tc.host.Host('http://127.0.0.1:8702')
ENDPOINT = '/transact/hypothetical'


class OperatorTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OperatorTests, self).__init__(*args, **kwargs)
        x = np.random.rand(2, 2)
        self.x_torch = torch.tensor(x, dtype=torch.float)
        w1 = np.random.rand(2, 2)
        self.w1_torch = torch.tensor(w1, dtype=torch.float, requires_grad=True)
        b1 = np.random.rand(2, 2)
        self.b1_torch = torch.tensor(b1, dtype=torch.float, requires_grad=True)
        w2 = np.random.rand(2, 2)
        self.w2_torch = torch.tensor(w2, dtype=torch.float, requires_grad=True)
        b2 = np.random.rand(2, 2)
        self.b2_torch = torch.tensor(b2, dtype=torch.float, requires_grad=True)
        self.x_tc = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.F32)
        self.w1_tc = tc.ml.optimizer.Variable.load(w1.shape, w1.flatten().tolist(), tc.F32)
        self.w2_tc = tc.ml.optimizer.Variable.load(w2.shape, w2.flatten().tolist(), tc.F32)
        self.b1_tc = tc.ml.optimizer.Variable.load(b1.shape, b1.flatten().tolist(), tc.F32)
        self.b2_tc = tc.ml.optimizer.Variable.load(b2.shape, b2.flatten().tolist(), tc.F32)

    def testAdd(self):
        y_torch = self.x_torch + self.w1_torch + self.b1_torch
        y2_torch = y_torch+self.w2_torch + self.b2_torch
        w1_torch_grad = grad(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc + self.w1_tc + self.b1_tc
        y_2tc = y_tc + self.w2_tc + self.b2_tc
        cxt.result = form_of(y_2tc).gradients(tc.tensor.Dense.ones(y_2tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testSub(self):
        y_torch = self.x_torch-self.w1_torch + self.b1_torch
        y2_torch = y_torch-self.w2_torch + self.b2_torch
        w1_torch_grad = grad(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc-self.w1_tc + self.b1_tc
        y_2tc = y_tc-self.w2_tc + self.b2_tc
        cxt.result = form_of(y_2tc).gradients(tc.tensor.Dense.ones(y_2tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testDiv(self):
        y_torch = self.x_torch/self.w1_torch + self.b1_torch
        y2_torch = y_torch/self.w2_torch + self.b2_torch
        w1_torch_grad = grad(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc/self.w1_tc + self.b1_tc
        y_2tc = y_tc/self.w2_tc + self.b2_tc
        cxt.result = form_of(y_2tc).gradients(tc.tensor.Dense.ones(y_2tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testPow(self):
        y_torch = self.x_torch**self.w1_torch + self.b1_torch
        y2_torch = y_torch**self.w2_torch + self.b2_torch
        w1_torch_grad = grad(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc**self.w1_tc + self.b1_tc
        y_2tc = y_tc**self.w2_tc + self.b2_tc
        cxt.result = form_of(y_2tc).gradients(tc.tensor.Dense.ones(y_2tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testMul(self):
        y_torch = self.x_torch*self.w1_torch + self.b1_torch
        y2_torch = y_torch*self.w2_torch + self.b2_torch
        w1_torch_grad = grad(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc*self.w1_tc + self.b1_tc
        y_2tc = y_tc*self.w2_tc + self.b2_tc
        cxt.result = form_of(y_2tc).gradients(tc.tensor.Dense.ones(y_2tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testMatMul(self):
        y_torch = self.x_torch@self.w1_torch + self.b1_torch
        y2_torch = y_torch@self.w2_torch + self.b2_torch
        w1_torch_grad = grad(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc@self.w1_tc + self.b1_tc
        y_2tc = y_tc@self.w2_tc + self.b2_tc
        cxt.result = form_of(y_2tc).gradients(tc.tensor.Dense.ones(y_2tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testExp(self):
        w_torch = self.w1_torch.exp()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.exp()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testLog(self):
        w_torch = self.w1_torch.log()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.log()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testSin(self):
        w_torch = self.w1_torch.sin()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.sin()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testCos(self):
        w_torch = self.w1_torch.cos()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.cos()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testAsin(self):
        w_torch = self.w1_torch.asin()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.asin()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testAcos(self):
        w_torch = self.w1_torch.acos()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.acos()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testSinh(self):
        w_torch = self.w1_torch.sinh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.sinh()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())    

    def testCosh(self):
        w_torch = self.w1_torch.cosh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.cosh()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testAsinh(self):
        w_torch = self.w1_torch.asinh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.asinh()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testAcosh(self):
        w_torch = (self.w1_torch*10).acosh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = (self.w1_tc*10).acosh()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testTan(self):
        w_torch = self.w1_torch.tan()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.tan()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testTanh(self):
        w_torch = self.w1_torch.tanh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.tanh()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testArctan(self):
        w_torch = self.w1_torch.atan()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.atan()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testArctanh(self):
        w_torch = self.w1_torch.atanh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.atanh()
        y_tc = self.x_tc*w_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testMultipleFunctions(self):
        y_torch = self.x_torch@self.w1_torch + self.w1_torch
        y2_torch = y_torch@self.w2_torch + self.b2_torch + torch.exp(y_torch)
        w1_torch_grad = grad(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc@self.w1_tc + self.w1_tc
        y_2tc = y_tc@self.w2_tc + self.b2_tc + y_tc.exp()
        cxt.result = form_of(y_2tc).gradients(tc.tensor.Dense.ones(y_2tc.shape))[hex_id(self.w1_tc)]
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))
        self.assertTrue((abs(w1_tc_grad-[t.detach().numpy() for t in w1_torch_grad]) < 0.0001).all())
    
    def testDerivative(self):
        y_torch = self.x_torch @ self.w1_torch + self.b1_torch + torch.exp(self.w1_torch)
        dy_dw1_torch = grad(y_torch,
                            self.w1_torch,
                            grad_outputs=torch.ones_like(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=torch.ones_like(dy_dw1_torch))[0]
        
        cxt = tc.Context()
        y_tc = self.x_tc @ self.w1_tc + self.b1_tc + self.w1_tc.exp()
        _dy_dw1_tc = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        _d2y_dw2_tc = form_of(_dy_dw1_tc).gradients(tc.tensor.Dense.ones(_dy_dw1_tc.shape))[hex_id(self.w1_tc)]
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})
        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = load_np(result['the_first_derivative'])
        d2y_dw2_tc = load_np(result['the_second_derivative'])

        self.assertTrue((abs(dy_dw1_tc-[t.detach().numpy() for t in dy_dw1_torch]) < 0.0001).all())
        self.assertTrue((abs(d2y_dw2_tc-[t.detach().numpy() for t in d2y_dw12_torch]) < 0.0001).all())


def load_np(as_json, dtype=float):
    shape = as_json[TENSOR_URI][0][0]
    return np.array(as_json[TENSOR_URI][1], dtype).reshape(shape)


if __name__ == "__main__":
    unittest.main()
