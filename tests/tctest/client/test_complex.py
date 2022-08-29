import tinychain as tc
import unittest

import numpy as np

from .base import ClientTest

ENDPOINT = "/transact/hypothetical"


# TODO: move to the host test package
class ComplexNumberOpsTests(ClientTest):
    def testReal(self):
        n = complex(1, 2)

        cxt = tc.Context()
        cxt.z = tc.C64(n).real

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, n.real)

    def testImag(self):
        n = complex(1, 2)

        cxt = tc.Context()
        cxt.z = tc.C64(n).imag

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, n.imag)

    def testAngle(self):
        n = 3 + 4j

        cxt = tc.Context()
        cxt.z = tc.C64(n).angle()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(actual, np.angle(n))

    def testConjugate(self):
        n = 2 - 2j

        cxt = tc.Context()
        cxt.z = tc.C64(n).conj()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), n.conjugate())

    def testAbs(self):
        n = complex(3, 4)

        cxt = tc.Context()
        cxt.z = tc.C64(n).abs()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, abs(n))

    def testSum(self):
        n = complex(4, 4)

        cxt = tc.Context()
        cxt.z = tc.C64(n) + tc.C64(n)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), n + n)

    def testSub(self):
        n = complex(4, 4)

        cxt = tc.Context()
        cxt.z = tc.C64(n) - tc.C64(n)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), n - n)

    def testProdComplex(self):
        a = complex(1, 2)
        b = complex(3, 4)

        cxt = tc.Context()
        cxt.z = tc.C64(a) * tc.C64(b)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), a * b)

    def testProdReal(self):
        a = complex(1, 2)
        b = 2

        cxt = tc.Context()
        cxt.z = tc.C64(a) * b

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), a * b)

    def testDivComplex(self):
        a = complex(4, 4)
        b = complex(2, 2)

        cxt = tc.Context()
        cxt.z = tc.C64(a) / tc.C64(b)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), a / b)

    def testDivReal1(self):
        a = complex(4, 4)
        b = 2

        cxt = tc.Context()
        cxt.z = tc.C64(a) / b

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), a / b)

    def testDivReal2(self):
        a = 2
        b = complex(2, 2)

        cxt = tc.Context()
        cxt.z = a / tc.C64(b)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(expect_number(actual), a / b)

    def testPow(self):
        n = 2.3 + 3.4j

        cxt = tc.Context()
        cxt.z = tc.C64(n)**3

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(expect_number(actual), n**3)

    def testExp(self):
        n = 0.25j

        cxt = tc.Context()
        cxt.z = (tc.C64(n) * np.pi).exp()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(expect_number(actual), np.exp(n * np.pi))


# TODO: move to test_tensor.py
class ComplexTensorOpsTests(ClientTest):
    def testCreateComplexTensor(self):
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x)

    def testSumComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) + tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x + n)

    def testSubComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) - tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x - n)

    def testProdComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) * tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x * n)

    def testDivComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) / tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, (x / n))

    def testSumComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) + n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x + n)

    def testSubComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) - n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x - n)

    def testProdComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) * n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x * n)

    def testDivComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) / n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, (x / n))

    def testSumComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2 + np.ones(shape)*2j
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) + load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 + x2)

    def testSubComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2 + np.ones(shape)*2j
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) - load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 - x2)

    def testProdComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2 + np.ones(shape)*2j
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) * load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 * x2)

    def testDivComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2 + np.ones(shape)*2j

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) / load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 / x2)

    def testSumComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) + load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 + x2)

    def testSubComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) - load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 - x2)

    def testProdComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) * load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 * x2)

    def testDivComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) / load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 / x2)

    def _expectDense(self, actual, expected):
        ((shape, dtype), actual) = actual[str(tc.URI(tc.tensor.Dense))]
        self.assertEqual(tuple(shape), expected.shape)
        actual = np.array([complex(actual[i], actual[i + 1]) for i in range(0, len(actual), 2)])
        self.assertTrue(np.allclose(expected.flatten(), actual))


def expect_number(as_json, dtype=tc.C64):
    return complex(*as_json[str(tc.URI(dtype))])


def load_dense(x, dtype=tc.C32):
    if issubclass(dtype, tc.Complex):
        data = [[n.real, n.imag] for n in x.flatten().tolist()]
    else:
        data = x.flatten().tolist()

    return tc.tensor.Dense.load(x.shape, data, dtype)


if __name__ == "__main__":
    unittest.main()
