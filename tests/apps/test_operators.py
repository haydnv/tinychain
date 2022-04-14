import numpy as np
import unittest
import testutils
import tinychain as tc
import torch
from torch.autograd import grad
from tinychain.util import form_of, hex_id

# LIB_URI = tc.uri(tc.math.linalg.LinearAlgebra)
TENSOR_URI = str(tc.uri(tc.tensor.Dense))
HOST = tc.host.Host('http://127.0.0.1:8702')
ENDPOINT = '/transact/hypothetical'


class OperatorTests(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     cls.host = testutils.start_host("test_neural_net", tc.math.linalg.LinearAlgebra(), wait_time=2)

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

    # TODO: make it works (chain rule)
    @unittest.skip
    def testMultipleFunctions(self):
        y_torch = self.x_torch@self.w1_torch + self.w1_torch
        print(f'y_torch: {y_torch}')
        # y2_torch = y_torch@self.w2_torch + self.b2_torch
        w1_torch_grad = grad(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        y_tc = self.x_tc@self.w1_tc + self.w1_tc
        # y_2tc = y_tc@self.w2_tc + self.b2_tc
        cxt.result = form_of(y_tc).gradients(tc.tensor.Dense.ones(y_tc.shape))[hex_id(self.w1_tc)]
        r1 = torch.transpose(self.x_torch, 1, 0)@torch.ones((2,2)) + 1
        print(r1)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))
        print(f'w1_tc_grad: {w1_tc_grad}')
        print(f'w1_torch_grad: {w1_torch_grad}')
        self.assertTrue((abs(w1_tc_grad-[t.detach().numpy() for t in w1_torch_grad]) < 0.0001).all())

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     cls.host.stop()


def load_np(as_json, dtype=float):
    shape = as_json[TENSOR_URI][0][0]
    return np.array(as_json[TENSOR_URI][1], dtype).reshape(shape)


if __name__ == "__main__":
    unittest.main()
