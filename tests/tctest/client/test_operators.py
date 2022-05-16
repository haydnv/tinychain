import math
import numpy as np
import unittest
import tinychain as tc
import torch

from torch.autograd import grad as grad_torch
from tinychain.collection.tensor import Dense
from tinychain.math.operator import gradients as grad_tc

TENSOR_URI = str(tc.uri(Dense))
HOST = tc.host.Host('http://127.0.0.1:8702')
ENDPOINT = '/transact/hypothetical'


ones_like_torch = torch.ones_like
ones_like_tc = tc.tensor.Dense.ones_like


# based on https://mathinsight.org/chain_rule_simple_examples
class ChainRuleTests(unittest.TestCase):
    def testAdd(self):
        cxt = tc.Context()
        cxt.x = tc.ml.Variable.ones([1])
        cxt.g_x = -2 * cxt.x + 5
        cxt.f_x = 6 * cxt.g_x + 3
        cxt.d_f_x = tc.math.derivative_of(cxt.f_x)
        cxt.f_x_grad = tc.math.gradients(cxt.f_x, ones_like_tc(cxt.f_x), cxt.x)
        cxt.passed = (cxt.d_f_x == cxt.f_x_grad).all()

        self.assertTrue(HOST.post(ENDPOINT, cxt))

    def testExp(self):
        cxt = tc.Context()
        cxt.x = tc.ml.Variable.ones([1])
        cxt.g_x = 4 * cxt.x
        cxt.h_x = cxt.g_x.exp()
        cxt.result = tc.math.derivative_of(cxt.h_x)

        expected = 4 * math.e**4
        actual = HOST.post(ENDPOINT, cxt)
        self.assertTrue(np.allclose(load_np(actual), np.array([expected])))


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
        y_tc = self.x_tc + self.w1_tc + self.b1_tc
        y_2tc = y_tc + self.w2_tc + self.b2_tc
        cxt.result = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testSub(self):
        y_torch = self.x_torch-self.w1_torch + self.b1_torch
        y2_torch = y_torch-self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc-self.w1_tc + self.b1_tc
        y_2tc = y_tc-self.w2_tc + self.b2_tc
        cxt.result = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testDiv(self):
        w1 = np.random.rand(2, 2) + 1
        self.w1_torch = torch.tensor(w1, dtype=torch.float, requires_grad=True)
        w2 = np.random.rand(2, 2) + 1
        self.w2_torch = torch.tensor(w2, dtype=torch.float, requires_grad=True)
        y_torch = self.x_torch/self.w1_torch + self.b1_torch
        y2_torch = y_torch/self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        self.w1_tc = tc.ml.optimizer.Variable.load(w1.shape, w1.flatten().tolist(), tc.F32)
        self.w2_tc = tc.ml.optimizer.Variable.load(w2.shape, w2.flatten().tolist(), tc.F32)
        y_tc = self.x_tc/self.w1_tc + self.b1_tc
        y_2tc = y_tc/self.w2_tc + self.b2_tc
        cxt.result = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testPow(self):
        y_torch = self.x_torch**self.w1_torch + self.b1_torch
        y2_torch = y_torch**self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc**self.w1_tc + self.b1_tc
        y_2tc = y_tc**self.w2_tc + self.b2_tc
        cxt.result = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testMul(self):
        y_torch = self.x_torch*self.w1_torch + self.b1_torch
        y2_torch = y_torch*self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc*self.w1_tc + self.b1_tc
        y_2tc = y_tc*self.w2_tc + self.b2_tc
        cxt.result = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testMatMul(self):
        y_torch = self.x_torch@self.w1_torch + self.b1_torch
        y2_torch = y_torch@self.w2_torch + self.b2_torch
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc@self.w1_tc + self.b1_tc
        y_2tc = y_tc@self.w2_tc + self.b2_tc
        cxt.result = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testExp(self):
        w_torch = self.w1_torch.exp()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.exp()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())


    def testLog(self):
        w_torch = self.w1_torch.log()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.log()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testSin(self):
        w_torch = self.w1_torch.sin()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.sin()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testCos(self):
        w_torch = self.w1_torch.cos()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.cos()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testAsin(self):
        w_torch = self.w1_torch.asin()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.asin()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testAcos(self):
        w_torch = self.w1_torch.acos()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.acos()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        response = HOST.post(ENDPOINT, cxt)
        w1_tc_grad = load_np(response)

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testSinh(self):
        w_torch = self.w1_torch.sinh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.sinh()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())    

    def testCosh(self):
        w_torch = self.w1_torch.cosh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.cosh()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testAsinh(self):
        w_torch = self.w1_torch.asinh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.asinh()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testAcosh(self):
        w1 = np.random.rand(2, 2)*10 + 1.1
        x = np.random.rand(2, 2) + 1
        self.w1_torch = torch.tensor(w1, dtype=torch.float, requires_grad=True)
        self.x_torch = torch.tensor(x, dtype=torch.float)
        y_torch = (self.x_torch*self.w1_torch).acosh()
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        self.w1_tc = tc.ml.optimizer.Variable.load(w1.shape, w1.flatten().tolist(), tc.F32)
        self.x_tc = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.F32)
        y_tc = (self.x_tc*self.w1_tc).acosh()
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testTan(self):
        w_torch = self.w1_torch.tan()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.tan()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testTanh(self):
        w_torch = self.w1_torch.tanh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.tanh()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testArctan(self):
        w_torch = self.w1_torch.atan()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.atan()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testArcTanh(self):
        w_torch = (self.w1_torch).atanh()
        y_torch = self.x_torch*w_torch
        w1_torch_grad = grad_torch(y_torch, self.w1_torch, grad_outputs=ones_like_torch(y_torch))

        cxt = tc.Context()
        w_tc = self.w1_tc.atanh()
        y_tc = self.x_tc*w_tc
        cxt.result = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    def testMultipleFunctions(self):
        y_torch = self.x_torch@self.w1_torch + self.w1_torch
        y2_torch = y_torch@self.w2_torch + self.b2_torch + torch.exp(y_torch)
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=ones_like_torch(y2_torch))

        cxt = tc.Context()
        y_tc = self.x_tc@self.w1_tc + self.w1_tc
        y_2tc = y_tc@self.w2_tc + self.b2_tc + y_tc.exp()
        cxt.result = grad_tc(y_2tc, ones_like_tc(y_2tc), self.w1_tc)
        w1_tc_grad = load_np(HOST.post(ENDPOINT, cxt))
        self.assertTrue((abs(w1_tc_grad-[t.detach().numpy() for t in w1_torch_grad]) < 0.0001).all())

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
        y_tc = self.x_tc @ self.w1_tc + self.b1_tc + self.w1_tc.exp()
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        _d2y_dw2_tc = grad_tc(_dy_dw1_tc, ones_like_tc(_dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})
        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = load_np(result['the_first_derivative'])
        d2y_dw2_tc = load_np(result['the_second_derivative'])

        self.assertTrue((abs(dy_dw1_tc-[t.detach().numpy() for t in dy_dw1_torch]) < 0.0001).all())
        self.assertTrue((abs(d2y_dw2_tc-[t.detach().numpy() for t in d2y_dw12_torch]) < 0.0001).all())
    
    def testSlice(self):
        y_torch = (self.x_torch @ self.w1_torch + self.b1_torch + torch.exp(self.w1_torch))*12
        dy_dw1_torch = grad_torch(y_torch,
                            self.w1_torch,
                            grad_outputs=ones_like_torch(y_torch),
                            create_graph=True,
                            retain_graph=True)[0]
        d2y_dw12_torch = grad_torch(dy_dw1_torch[..., 1],
                              self.w1_torch,
                              grad_outputs=ones_like_torch(dy_dw1_torch[..., 1]))[0]
        cxt = tc.Context()
        y_tc = (self.x_tc @ self.w1_tc + self.b1_tc + self.w1_tc.exp())*12
        _dy_dw1_tc = grad_tc(y_tc, ones_like_tc(y_tc), self.w1_tc)
        __dy_dw1_tc = _dy_dw1_tc[:, 1]
        _d2y_dw2_tc = grad_tc(__dy_dw1_tc, ones_like_tc(__dy_dw1_tc), self.w1_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})
        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = load_np(result['the_first_derivative'])
        d2y_dw2_tc = load_np(result['the_second_derivative'])
        self.assertTrue((abs(dy_dw1_tc-[t.detach().numpy() for t in dy_dw1_torch]) < 0.0001).all())
        self.assertTrue((abs(d2y_dw2_tc-[t.detach().numpy() for t in d2y_dw12_torch]) < 0.0001).all())

    def testConcat(self):
        y_torch = torch.cat([self.w1_torch, self.b1_torch])

        torch_grad = grad_torch(
            y_torch, [self.w1_torch, self.b1_torch], grad_outputs=torch.ones_like(y_torch))

        cxt = tc.Context()
        y_tc = tc.tensor.Dense.concatenate([self.w1_tc, self.b1_tc])
        cxt.result = tc.math.operator.operator(y_tc).gradients(tc.tensor.Dense.ones([4, 2]))

        tc_grad = HOST.post(ENDPOINT, cxt)

        self.assertEqual(len(torch_grad), len(tc_grad))
        for (expected, actual) in zip(torch_grad, tc_grad.values()):
            actual = load_np(actual)
            self.assertTrue((abs(actual - expected.numpy()) < 0.0001).all())

    @unittest.skip  # TODO: why doesn't this work?
    def testSum_gradient(self):
        y_torch = (self.x_torch @ torch.exp(self.w1_torch) + self.b1_torch)**2
        y2_torch = torch.sum(y_torch, 0) ** 0.5
        w1_torch_grad = grad_torch(y2_torch, self.w1_torch, grad_outputs=torch.ones_like(y2_torch))

        cxt = tc.Context()
        cxt.y_tc = (self.x_tc @ self.w1_tc.exp() + self.b1_tc)**2
        cxt.y_2tc = cxt.y_tc.sum(0)**0.5
        cxt.result = grad_tc(cxt.y_2tc, ones_like_tc(cxt.y_2tc), self.w1_tc)

        response = HOST.post(ENDPOINT, cxt)
        w1_tc_grad = load_np(response)

        self.assertTrue((abs(w1_tc_grad-[t.numpy() for t in w1_torch_grad]) < 0.0001).all())

    @unittest.skip # TODO: handle chain rule correctly
    def testSum_derivative(self):
        # based on https://math.stackexchange.com/questions/289989/first-and-second-derivative-of-a-summation

        from tinychain.math.operator import derivative_of

        n = 10
        mu = np.array([3])
        x = np.arange(n).reshape([n])

        cxt = tc.Context()
        cxt.mu = tc.ml.Variable.load(shape=[1], data=[3])
        cxt.x = tc.tensor.Dense.arange([n], 0, n)
        cxt.f_x = ((cxt.x - cxt.mu)**2).sum()
        cxt.d_f_x = derivative_of(cxt.f_x)
        cxt.d2_f_x = derivative_of(cxt.d_f_x)
        cxt.result = cxt.d_f_x, cxt.d2_f_x

        expected_d = -2 * np.sum(x - mu)
        expected_d2 = 2 * n

        self.assertEqual(HOST.post(ENDPOINT, cxt), [expected_d, expected_d2])

    @unittest.skip # TODO: make it work
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
        print(_d2y_dw2_tc)
        cxt.map = tc.Map({'the_first_derivative': _dy_dw1_tc, 'the_second_derivative': _d2y_dw2_tc})
        result = HOST.post(ENDPOINT, cxt)
        dy_dw1_tc = load_np(result['the_first_derivative'])
        d2y_dw2_tc = load_np(result['the_second_derivative'])

        self.assertTrue((abs(dy_dw1_tc-[t.detach().numpy() for t in dy_dw1_torch]) < 0.0001).all())
        self.assertTrue((abs(d2y_dw2_tc-[t.detach().numpy() for t in d2y_dw12_torch]) < 0.0001).all())


def load_np(as_json, dtype=float):
    shape = as_json[TENSOR_URI][0][0]
    ndarray = np.array(as_json[TENSOR_URI][1], dtype)
    return ndarray.reshape(shape)

if __name__ == "__main__":
    unittest.main()