import tinychain as tc
import unittest

import numpy as np

from base import ClientTest

ENDPOINT = "/transact/hypothetical"
C64_URI = str(tc.URI(tc.C64))


# TODO: fix failing tests
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

    # TODO: why does np.angle(a + bj) differ from np.arctan(a / b) in some cases, like 3 + 4j?
    def testAngle(self):
        n = 1 + 1j

        cxt = tc.Context()
        cxt.z = tc.C64(n).angle()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(actual, np.angle(n))

    def testConjugate(self):
        n = 2 - 2j

        cxt = tc.Context()
        cxt.z = tc.C64(n).conj()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(parse(actual), n.conjugate())

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
        self.assertEqual(parse(actual), n + n)

    def testSub(self):
        n = complex(4, 4)

        cxt = tc.Context()
        cxt.z = tc.C64(n) - tc.C64(n)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(parse(actual), n - n)

    def testProdComplex(self):
        a = complex(1, 2)
        b = complex(3, 4)

        cxt = tc.Context()
        cxt.z = tc.C64(a) * tc.C64(b)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(parse(actual), a * b)

    def testProdReal(self):
        a = complex(1, 2)
        b = 2

        cxt = tc.Context()
        cxt.z = tc.C64(a) * b

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(parse(actual), a * b)

    def testDivComplex(self):
        a = complex(4, 4)
        b = complex(2, 2)

        cxt = tc.Context()
        cxt.z = tc.C64(a) / tc.C64(b)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(parse(actual), a / b)

    def testDivReal1(self):
        a = complex(4, 4)
        b = 2

        cxt = tc.Context()
        cxt.z = tc.C64(a) / b

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(parse(actual), a / b)

    def testDivReal2(self):
        a = 2
        b = complex(2, 2)

        cxt = tc.Context()
        cxt.z = a / tc.C64(b)

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(parse(actual), a / b)

    def testPow(self):
        n = 2.3 + 3.4j

        cxt = tc.Context()
        cxt.z = tc.C64(n)**3

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(parse(actual), n**3)

    def testExp(self):
        n = 0.25j

        cxt = tc.Context()
        cxt.z = (tc.C64(n) * np.pi).exp()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(parse(actual), np.exp(n * np.pi))


class ComplexTensorOpsTests(ClientTest):
    def testCreateComplexTensor(self):
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), x.flatten())
        self.assertEqual(actual, expected)

    def testSumComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) + tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x + n).flatten())
        self.assertEqual(actual, expected)

    def testSubComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) - tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x - n).flatten())
        self.assertEqual(actual, expected)

    def testProdComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) * tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x * n).flatten())
        self.assertEqual(actual, expected)

    def testDivComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) / tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x / n).flatten())
        self.assertEqual(actual, expected)

    def testSumComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) + n
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x + n).flatten())
        self.assertEqual(actual, expected)

    def testSubComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) - n
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x - n).flatten())
        self.assertEqual(actual, expected)

    def testProdComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) * n
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x * n).flatten())
        self.assertEqual(actual, expected)

    def testDivComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape)*1j
        
        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) / n
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x / n).flatten())
        self.assertEqual(actual, expected)

    def testSumComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2 + np.ones(shape)*2j
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) + load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x1 + x2).flatten())
        self.assertEqual(actual, expected)

    def testSubComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2 + np.ones(shape)*2j
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) - load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x1 - x2).flatten())
        self.assertEqual(actual, expected)

    def testProdComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2 + np.ones(shape)*2j
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) * load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x1 * x2).flatten())
        self.assertEqual(actual, expected)

    def testDivComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2 + np.ones(shape)*2j
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) / load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x1 / x2).flatten())
        self.assertEqual(actual, expected)

    def testSumComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) + load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x1 + x2).flatten())
        self.assertEqual(actual, expected)

    def testSubComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) - load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x1 - x2).flatten())
        self.assertEqual(actual, expected)

    def testProdComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) * load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x1 * x2).flatten())
        self.assertEqual(actual, expected)

    def testDivComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape)*1j
        x2 = np.ones(shape)*2
        
        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) / load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        expected = expect_dense(tc.C64, list(x.shape), (x1 / x2).flatten())
        self.assertEqual(actual, expected)


def parse(as_json):
    return complex(*as_json[C64_URI])


def expect_dense(dtype, shape, flat):
    return {
        str(tc.URI(tc.tensor.Dense)): [
            [list(shape), str(tc.URI(dtype))],
            list(flat),
        ]
    }


def load_dense(x, dtype=tc.F32):
    return tc.tensor.Dense.load(x.shape, x.flatten().tolist(), dtype)


if __name__ == "__main__":
    unittest.main()
