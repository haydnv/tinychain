import math
import numpy as np
import unittest
import tinychain as tc
import torch

from torch.autograd import grad as grad_torch
from tinychain.collection.tensor import Dense
from tinychain.math.operator import gradients as grad_tc

TENSOR_URI = str(tc.URI(Dense))
HOST = tc.host.Host('http://127.0.0.1:8702')
ENDPOINT = '/transact/hypothetical'


ones_like_torch = torch.ones_like
ones_like_tc = tc.tensor.Dense.ones_like


# based on:
#  - https://mathinsight.org/chain_rule_simple_examples
#  - https://math.hmc.edu/calculus/hmc-mathematics-calculus-online-tutorials/multivariable-calculus/multi-variable-chain-rule/
class ChainRuleTests(unittest.TestCase):
    def testAdd(self):
        cxt = tc.Context()
        cxt.x = tc.ml.Variable.ones([1])
        cxt.g_x = -2 * cxt.x + 5
        cxt.f_x = 6 * cxt.g_x + 3
        cxt.d_f_x = tc.math.derivative_of(cxt.f_x)
        cxt.f_x_grad = tc.math.gradients(cxt.f_x, ones_like_tc(cxt.f_x), cxt.x)
        cxt.result = (cxt.f_x_grad == cxt.d_f_x).all()

        passed = HOST.post(ENDPOINT, cxt)
        self.assertTrue(passed)

    def testExp_simple(self):
        cxt = tc.Context()
        cxt.x = tc.ml.Variable.ones([1])
        cxt.g_x = 4 * cxt.x
        cxt.h_x = cxt.g_x.exp()
        cxt.result = tc.math.derivative_of(cxt.h_x)

        expected = 4 * math.e**4
        actual = HOST.post(ENDPOINT, cxt)
        self.assertTrue(np.allclose(load_np(actual), np.array([expected])))

    def testExp_withOperatorExponent(self):
        cxt = tc.Context()
        cxt.x = tc.ml.Variable.ones([1])
        cxt.f_g_x = (3 * (cxt.x**2) + 2).exp()
        cxt.result = tc.math.derivative_of(cxt.f_g_x)

        x = np.array([1])
        expected = 6 * x * math.e**(3 * x**2 + 2)
        actual = HOST.post(ENDPOINT, cxt)

        self.assertTrue(np.allclose(load_np(actual), expected))

    def testLog(self):
        cxt = tc.Context()
        cxt.x = tc.ml.Variable.ones([1])
        cxt.g_x = (cxt.x**2 + 1).log()
        cxt.result = tc.math.derivative_of(cxt.g_x)

        x = np.array([1])
        expected = (2 * x) / (x**2 + 1)
        actual = HOST.post(ENDPOINT, cxt)

        self.assertTrue(np.allclose(load_np(actual), expected))

    def testMultipleVariables(self):
        cxt = tc.Context()
        cxt.t = tc.ml.Variable.ones([1])
        cxt.x = cxt.t**2
        cxt.y = 2*cxt.t
        cxt.z = (cxt.x**2 * cxt.y) - cxt.y**2
        cxt.result = tc.math.derivative_of(cxt.z)

        t = np.array([1])
        expected = (10 * t**4) - (8 * t)
        actual = HOST.post(ENDPOINT, cxt)

        self.assertTrue(np.allclose(load_np(actual), expected))


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
        self.x_tc = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.F32, name="x")
        self.w1_tc = tc.ml.optimizer.Variable.load(w1.shape, w1.flatten().tolist(), tc.F32, name="w1")
        self.w2_tc = tc.ml.optimizer.Variable.load(w2.shape, w2.flatten().tolist(), tc.F32, name="w2")
        self.b1_tc = tc.ml.optimizer.Variable.load(b1.shape, b1.flatten().tolist(), tc.F32, name="b1")
        self.b2_tc = tc.ml.optimizer.Variable.load(b2.shape, b2.flatten().tolist(), tc.F32, name="b2")

    def testAdd(self):
        y_torch = self.x_torch + self.w1_torch + self.b1_torch
        y2_torch = y_torch+self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        cxt.y_tc = self.x_tc + self.w1_tc + self.b1_tc
        cxt.y_2tc = cxt.y_tc + self.w2_tc + self.b2_tc
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testAdd2ndDerivative(self):
        y_torch = (self.x_torch + self.w1_torch)**2 + self.b1_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        y_tc = (self.x_tc + self.w1_tc)**2 + self.b1_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testSub(self):
        y_torch = self.x_torch-self.w1_torch + self.b1_torch
        y2_torch = y_torch-self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        cxt.y_tc = self.x_tc - self.w1_tc + self.b1_tc
        cxt.y_2tc = cxt.y_tc - self.w2_tc + self.b2_tc
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testSub2ndDerivative(self):
        y_torch = (self.x_torch - self.w1_torch)**2 + self.b1_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        y_tc = (self.x_tc - self.w1_tc)**2 + self.b1_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testDiv(self):
        w1 = np.random.rand(2, 2) + 1
        w1_torch = torch.tensor(w1, dtype=torch.float, requires_grad=True)
        w2 = np.random.rand(2, 2) + 1
        w2_torch = torch.tensor(w2, dtype=torch.float, requires_grad=True)

        y_torch = self.x_torch / w1_torch + self.b1_torch
        y2_torch = y_torch / w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, w1_torch, grad_outputs=ones_like_torch(y2_torch))

        w1_tc = tc.ml.optimizer.Variable.load(w1.shape, w1.flatten().tolist(), tc.F32)
        w2_tc = tc.ml.optimizer.Variable.load(w2.shape, w2.flatten().tolist(), tc.F32)

        cxt = tc.Context()
        cxt.y_tc = self.x_tc / w1_tc + self.b1_tc
        cxt.y_2tc = cxt.y_tc / w2_tc + self.b2_tc
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testDiv2ndDerivative(self):
        w1 = np.random.rand(2, 2) + 1
        self.w1_torch = torch.tensor(w1, dtype=torch.float, requires_grad=True)
        y_torch = self.x_torch/self.w1_torch + self.b1_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        self.w1_tc = tc.ml.optimizer.Variable.load(w1.shape, w1.flatten().tolist(), tc.F32)
        y_tc = self.x_tc/self.w1_tc + self.b1_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testPow1stDerivative(self):
        y_torch = self.x_torch**self.w1_torch + self.b1_torch
        y2_torch = y_torch**self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        cxt.y_tc = self.x_tc**self.w1_tc + self.b1_tc
        cxt.y_2tc = cxt.y_tc**self.w2_tc + self.b2_tc
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testPow2ndDerivative(self):
        y_torch = self.x_torch**self.w1_torch + self.b1_torch
        y2_torch = (y_torch**self.w2_torch + self.b2_torch)**5
        dy_dw1_torch = grad_torch(y2_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y2_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        y_tc = self.x_tc**self.w1_tc + self.b1_tc
        y_2tc = (y_tc**self.w2_tc + self.b2_tc)**5
        _dy_dw1_tc = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testMul(self):
        y_torch = self.x_torch * self.w1_torch + self.b1_torch
        y2_torch = y_torch * self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        cxt.y_tc = self.x_tc * self.w1_tc + self.b1_tc
        cxt.y_2tc = cxt.y_tc * self.w2_tc + self.b2_tc
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testMul2ndDerivative(self):
        y_torch = self.x_torch*self.w1_torch + self.b1_torch
        y2_torch = (y_torch*self.w2_torch + self.b2_torch) * self.x_torch * self.w1_torch
        dy_dw1_torch = grad_torch(y2_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y2_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        y_tc = self.x_tc*self.w1_tc + self.b1_tc
        y_2tc = (y_tc*self.w2_tc + self.b2_tc) * self.x_tc * self.w1_tc
        _dy_dw1_tc = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testMatMul(self):
        y_torch = self.x_torch@self.w1_torch + self.b1_torch
        y2_torch = y_torch@self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        cxt.y_tc = self.x_tc@self.w1_tc + self.b1_tc
        cxt.y_2tc = cxt.y_tc@self.w2_tc + self.b2_tc
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testMatMul2ndDerivative(self):
        y_torch = self.x_torch@self.w1_torch**2 + self.b1_torch
        y2_torch = (y_torch@self.w2_torch + self.b2_torch)**2
        dy_dw1_torch = grad_torch(y2_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y2_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        y_tc = self.x_tc@self.w1_tc**2 + self.b1_tc
        y_2tc = (y_tc@self.w2_tc + self.b2_tc)**2
        _dy_dw1_tc = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testExp(self):
        w_torch = self.w1_torch.exp()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.exp()
        cxt.y_tc = self.x_tc*cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testExp2ndDerivative(self):
        w_torch = self.w1_torch.exp()
        y_torch = self.x_torch*w_torch
        y2_torch = (y_torch * self.w1_torch).exp()
        dy_dw1_torch = grad_torch(y2_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y2_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.exp()
        y_tc = self.x_tc*w_tc
        y_2tc = (y_tc * self.w1_tc).exp()
        _dy_dw1_tc = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testLog(self):
        w_torch = self.w1_torch.log()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.log()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testLog2ndDerivative(self):
        w_torch = self.w1_torch.log()
        y_torch = self.x_torch*w_torch
        y2_torch = (y_torch * self.w1_torch).log()
        dy_dw1_torch = grad_torch(y2_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y2_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.log()
        y_tc = self.x_tc*w_tc
        y_2tc = (y_tc * self.w1_tc).log()
        _dy_dw1_tc = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testSin(self):
        w_torch = self.w1_torch.sin()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.sin()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testSin2ndDerivative(self):
        w_torch = self.w1_torch.sin()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.sin()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testCos(self):
        w_torch = self.w1_torch.cos()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.cos()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testCos2ndDerivative(self):
        w_torch = self.w1_torch.cos()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.cos()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testAsin(self):
        w_torch = self.w1_torch.asin()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.asin()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testAsin2ndDerivative(self):
        w_torch = self.w1_torch.asin()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.asin()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testAcos(self):
        w_torch = self.w1_torch.acos()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.acos()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testAcos2ndDerivative(self):
        w_torch = self.w1_torch.acos()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.acos()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testSinh(self):
        w_torch = self.w1_torch.sinh()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.sinh()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)
    
    def testSinh2ndDerivative(self):
        w_torch = self.w1_torch.sinh()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.sinh()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testCosh(self):
        w_torch = self.w1_torch.cosh()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.cosh()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testCosh2ndDerivative(self):
        w_torch = self.w1_torch.cosh()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.cosh()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testAsinh(self):
        w_torch = self.w1_torch.asinh()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.asinh()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)
    
    def testAsinh2ndDerivative(self):
        w_torch = self.w1_torch.asinh()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.asinh()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testAcosh(self):
        w1 = np.random.rand(2, 2)*10 + 1.1
        x = np.random.rand(2, 2) + 1

        w1_torch = torch.tensor(w1, dtype=torch.float, requires_grad=True)
        x_torch = torch.tensor(x, dtype=torch.float)
        y_torch = (x_torch * w1_torch).acosh()
        w1_torch_grad = grad_torch(y_torch, w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        cxt.w1_tc = tc.ml.optimizer.Variable.load(w1.shape, w1.flatten().tolist(), tc.F32)
        cxt.x_tc = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.F32)
        cxt.y_tc = (cxt.x_tc * cxt.w1_tc).acosh()
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), cxt.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)
    
    def testAcosh2ndDerivative(self):
        w1 = np.random.rand(2, 2)*10 + 1.1
        x = np.random.rand(2, 2) + 1
        self.w1_torch = torch.tensor(w1, dtype=torch.float, requires_grad=True)
        self.x_torch = torch.tensor(x, dtype=torch.float)
        y_torch = (self.x_torch*self.w1_torch).acosh()
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        self.w1_tc = tc.ml.optimizer.Variable.load(w1.shape, w1.flatten().tolist(), tc.F32)
        self.x_tc = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.F32)
        y_tc = (self.x_tc*self.w1_tc).acosh()
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testTan(self):
        w_torch = self.w1_torch.tan()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.tan()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testTan2ndDerivative(self):
        w_torch = self.w1_torch.tan()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.tan()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testTanh(self):
        w_torch = self.w1_torch.tanh()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.tanh()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testTanh2ndDerivative(self):
        w_torch = self.w1_torch.tanh()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.tanh()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testAtan(self):
        w_torch = self.w1_torch.atan()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.atan()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testAtan2ndDerivative(self):
        w_torch = self.w1_torch.atan()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.atan()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testAtanh(self):
        w_torch = (self.w1_torch).atanh()
        y_torch = self.x_torch * w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        cxt.w_tc = self.w1_tc.atanh()
        cxt.y_tc = self.x_tc * cxt.w_tc
        cxt.result = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testAtanh2ndDerivative(self):
        w_torch = self.w1_torch.atanh()
        y_torch = self.x_torch*w_torch
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        w_tc = self.w1_tc.atanh()
        y_tc = self.x_tc*w_tc
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc, 0.01)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc, 0.01)

    def testMultipleFunctions(self):
        y_torch = self.x_torch @ self.w1_torch + self.w1_torch
        y2_torch = y_torch @ self.w2_torch + self.b2_torch + torch.exp(y_torch)
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        cxt.y_tc = self.x_tc @ self.w1_tc + self.w1_tc
        cxt.y_2tc = cxt.y_tc @ self.w2_tc + self.b2_tc + cxt.y_tc.exp()
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testDerivative(self):
        y_torch = self.x_torch @ self.w1_torch + self.b1_torch + torch.exp(self.w1_torch)
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch))[0]

        cxt = tc.Context()
        cxt.y_tc = self.x_tc @ self.w1_tc + self.b1_tc + self.w1_tc.exp()
        cxt._dy_dw1_tc = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)
        cxt._d2y_dw2_tc = grad_tc(cxt._dy_dw1_tc, ones_like_tc(cxt._dy_dw1_tc), self.w1_tc)
        cxt.result = {'the_first_derivative': cxt._dy_dw1_tc, 'the_second_derivative': cxt._d2y_dw2_tc}

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    # TODO: test the second derivative of a slice
    def testSlice(self):
        y_torch = (self.x_torch @ self.w1_torch + self.b1_torch + torch.exp(self.w1_torch)) * 12
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch[..., 1],
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch[..., 1]))[0]

        cxt = tc.Context()
        cxt.y_tc = (self.x_tc @ self.w1_tc + self.b1_tc + self.w1_tc.exp()) * 12
        cxt._dy_dw1_tc = grad_tc(cxt.y_tc, ones_like_tc(cxt.y_tc), self.w1_tc)
        cxt.__dy_dw1_tc = cxt._dy_dw1_tc[:, 1]
        cxt._d2y_dw2_tc = grad_tc(cxt.__dy_dw1_tc, ones_like_tc(cxt.__dy_dw1_tc), self.w1_tc)
        cxt.result = {'the_first_derivative': cxt._dy_dw1_tc, 'the_second_derivative': cxt._d2y_dw2_tc}

        result = HOST.post(ENDPOINT, cxt)

        dy_dw1_tc = result['the_first_derivative']
        # d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        # self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def testConcat(self):
        y_torch = torch.cat([self.w1_torch, self.b1_torch])

        torch_grad = grad_torch(
            y_torch, [self.w1_torch, self.b1_torch], grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        cxt.y_tc = tc.tensor.Dense.concatenate([self.w1_tc, self.b1_tc])
        cxt.result = tc.math.operator.operator(cxt.y_tc).gradients(ones_like_tc(cxt.y_tc))

        tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertEqual(len(torch_grad), len(tc_grad))
        for (expected, actual) in zip(torch_grad, tc_grad.values()):
            self.assertAllClose(expected, actual)

    def testSum_derivative(self):
        # based on https://math.stackexchange.com/questions/289989/first-and-second-derivative-of-a-summation

        from tinychain.math.operator import derivative_of

        n = 10
        mu = np.array([3])
        x = np.arange(n).reshape([n])

        cxt = tc.Context()
        cxt.mu = tc.ml.Variable.load(shape=[1], data=[3], name="mu")
        cxt.x = tc.tensor.Dense.arange([n], 0, n, name="x")
        cxt.f_x = ((cxt.x - cxt.mu)**2).sum()
        cxt.d_f_x = derivative_of(cxt.f_x)
        cxt.d2_f_x = derivative_of(cxt.d_f_x)
        cxt.result = cxt.d_f_x, cxt.d2_f_x

        expected_d = -2 * np.sum(x - mu)
        expected_d2 = 2 * n

        actual_d, actual_d2 = HOST.post(ENDPOINT, cxt)
        self.assertEqual(actual_d, expected_d)
        self.assertEqual(actual_d2, expected_d2)

    def testSum_gradient(self):
        y_torch = (self.x_torch @ torch.exp(self.w1_torch) + self.b1_torch)**2
        y2_torch = torch.sum(y_torch, 0)**0.5
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        cxt.y_tc = (self.x_tc @ self.w1_tc.exp() + self.b1_tc)**2
        cxt.y_2tc = cxt.y_tc.sum(0)**0.5
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), self.w1_tc)

        w1_tc_grad = HOST.post(ENDPOINT, cxt)
        self.assertAllClose(w1_torch_grad, w1_tc_grad)

    def testSum2ndDerivative(self):
        y_torch = (self.x_torch @ self.w1_torch + self.b1_torch)**2
        y2_torch = torch.sum(y_torch, 0)**0.5
        dy_dw1_torch = grad_torch(y2_torch,
                                  self.w1_torch,
                                  grad_outputs=torch.ones_like(y2_torch),
                                  create_graph=True,
                                  retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch,
                                    self.w1_torch,
                                    grad_outputs=torch.ones_like(dy_dw1_torch))[0]

        cxt = tc.Context()
        y_tc = (self.x_tc @ self.w1_tc + self.b1_tc)**2
        y_2tc = y_tc.sum(0)**0.5
        _dy_dw1_tc = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})

        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = result['the_first_derivative']
        d2y_dw2_tc = result['the_second_derivative']

        self.assertAllClose(dy_dw1_torch, dy_dw1_tc)
        self.assertAllClose(d2y_dw12_torch, d2y_dw2_tc)

    def assertAllClose(self, tensor_torch, tensor_tc, threshold=0.0001):
        self.assertTrue((abs(load_np(tensor_tc) - [t.detach().numpy() for t in tensor_torch]) < threshold).all())


def load_np(as_json, dtype=float):
    shape = as_json[TENSOR_URI][0][0]
    ndarray = np.array(as_json[TENSOR_URI][1], dtype)
    return ndarray.reshape(shape)

if __name__ == "__main__":
    unittest.main()