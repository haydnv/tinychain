import numpy as np
import tinychain as tc
import unittest

from .base import HostTest


ENDPOINT = "/transact/hypothetical"


class ComplexNumberTests(HostTest):
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
        cxt.n = n
        cxt.z = tc.C64(tc.URI('n')) - cxt.n

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), n - n)

    def testProdComplex(self):
        a = complex(1, 2)
        b = complex(3, 4)

        cxt = tc.Context()
        cxt.a = complex(1, 2)
        cxt.b = complex(3, 4)
        cxt.z1 = cxt.a * cxt.b
        cxt.z2 = tc.Complex(tc.URI('a')) * cxt.b
        cxt.check = cxt.z1 == cxt.z2

        self.assertTrue(self.host.post(ENDPOINT, cxt))

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
        cxt = tc.Context()
        cxt.a = 2
        cxt.b = complex(2, 2)
        cxt.z1 = cxt.a / cxt.b
        cxt.z2 = cxt.a / tc.URI('b')
        cxt.check = cxt.z1 == cxt.z2

        self.assertTrue(self.host.post(ENDPOINT, cxt))

    def testPow(self):
        cxt = tc.Context()
        cxt.n = 2.3 + 3.4j
        cxt.z1 = cxt.n ** 3
        cxt.z2 = tc.C32(tc.URI('n')) ** 3
        cxt.test = cxt.z1 == cxt.z2

        self.assertTrue(self.host.post(ENDPOINT, cxt))

    def testExp(self):
        cxt = tc.Context()
        cxt.n = 0.25j
        cxt.z1 = (cxt.n * np.pi).exp()
        cxt.z2 = (tc.C64(tc.URI('n')) * np.pi).exp()
        cxt.test = cxt.z1 == cxt.z2

        self.assertTrue(self.host.post(ENDPOINT, cxt))


class ComplexDenseTests(HostTest):
    def testCreateComplexTensor(self):
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x)

    def testSumComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) + tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x + n)

    def testSubComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) - tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x - n)

    def testProdComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) * tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x * n)

    def testDivComplexTensorComplexNumber(self):
        n = 2 + 5j
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) / tc.C64(n)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, (x / n))

    def testMulRealTensorComplexNumber(self):
        n = 2 + 2j
        shape = (3, 6)
        x = np.ones(shape)

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.F64) * n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x * n)

    def testDivRealTensorComplexNumber(self):
        n = 2 + 2j
        shape = (3, 6)
        x = np.ones(shape)

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.F32) / n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, (x / n))

    def testAddComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) + n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x + n)

    def testSubComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) - n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x - n)

    def testProdComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) * n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x * n)

    def testDivComplexTensorRealNumber(self):
        n = 2
        shape = (3, 6)
        x = np.ones(shape) + np.ones(shape) * 1j

        cxt = tc.Context()
        cxt.x = load_dense(x, tc.C64) / n
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, (x / n))

    def testAddComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape) * 1j
        x2 = np.ones(shape) * 2 + np.ones(shape) * 2j

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) + load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 + x2)

    def testSubComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape) * 1j
        x2 = np.ones(shape) * 2 + np.ones(shape) * 2j

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) - load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 - x2)

    def testProdComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape) * 1j
        x2 = np.ones(shape) * 2 + np.ones(shape) * 2j

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) * load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 * x2)

    def testDivComplexTensorComplexTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape) * 1j
        x2 = np.ones(shape) * 2 + np.ones(shape) * 2j

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) / load_dense(x2, tc.C64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 / x2)

    def testAddComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape) * 1j
        x2 = np.ones(shape) * 2

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) + load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 + x2)

    def testSubComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape) * 1j
        x2 = np.ones(shape) * 2

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) - load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 - x2)

    def testMulComplexNumberRealTensor(self):
        n = 10
        x1 = 2j * np.arange(n)

        cxt = tc.Context()
        cxt.x = (tc.C32(2j) * tc.tensor.Dense.arange([n], 0., float(n)))
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1)

    def testMulComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape) * 1j
        x2 = np.ones(shape) * 2

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) * load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 * x2)

    def testDivComplexTensorRealTensor(self):
        shape = (3, 6)
        x1 = np.ones(shape) + np.ones(shape) * 1j
        x2 = np.ones(shape) * 2

        cxt = tc.Context()
        cxt.x = load_dense(x1, tc.C64) / load_dense(x2, tc.I64)
        actual = self.host.post(ENDPOINT, cxt)
        self._expectDense(actual, x1 / x2)

    def _expectDense(self, actual, expected):
        ((dtype, shape), actual) = actual[str(tc.URI(tc.tensor.Dense))]
        self.assertEqual(tuple(shape), expected.shape)

        actual = np.array([complex(actual[i], actual[i + 1]) for i in range(0, len(actual), 2)])
        passed = np.allclose(expected.flatten(), actual)
        self.assertTrue(passed)


class ComplexSparseTests(HostTest):
    def testCreateComplexTensor(self):
        shape = (3, 6)
        coord = [0, 0]
        value = tc.C32(3 + 4j)

        cxt = tc.Context()
        cxt.tensor = tc.tensor.Sparse.zeros(shape, tc.C32)
        cxt.result = tc.after(cxt.tensor[coord].write(value), cxt.tensor)
        actual = self.host.post(ENDPOINT, cxt)

        expected = np.zeros(shape, dtype=complex)
        expected[tuple(coord)] = 3 + 4j
        self._expectSparse(actual, expected)

    def testSumComplexTensorComplexTensor(self):
        shape = [5, 2, 3]

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape, tc.C32)
        cxt.small = tc.tensor.Sparse.zeros([3], tc.C32)
        cxt.result = tc.after([
            cxt.big[1, 0, 0].write(tc.C32(1 + 1j)),
            cxt.small[1].write(tc.C32(2 + 2j)),
        ], cxt.big + cxt.small)

        actual = self.host.post(ENDPOINT, cxt)

        big = np.zeros(shape, dtype=complex)
        big[1, 0, 0] = 1 + 1j
        small = np.zeros([3], dtype=complex)
        small[1] = 2 + 2j
        expected = big + small

        self._expectSparse(actual, expected)

    def testSumComplexTensorRealTensor(self):
        shape = [5, 2, 3]

        cxt = tc.Context()
        cxt.big = tc.tensor.Sparse.zeros(shape, tc.C32)
        cxt.small = tc.tensor.Sparse.zeros([3], tc.F32)
        cxt.result = tc.after([
            cxt.big[1, 0, 0].write(tc.C32(1 + 1j)),
            cxt.small[1].write(2),
        ], cxt.big + cxt.small)

        actual = self.host.post(ENDPOINT, cxt)

        big = np.zeros(shape, dtype=complex)
        big[1, 0, 0] = 1 + 1j
        small = np.zeros([3])
        small[1] = 2
        expected = big + small

        self._expectSparse(actual, expected)

    def testDivComplexTensorRealTensor(self):
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([30, 3, 2], 1., 181.)
        cxt.sparse = tc.tensor.Sparse.zeros([3, 2], tc.C32)
        cxt.result = tc.after(cxt.sparse[1, 0].write(tc.C32(2 + 2j)), cxt.sparse / cxt.dense)

        actual = self.host.post(ENDPOINT, cxt)
        l = np.arange(1, 181).reshape([30, 3, 2])
        r = np.zeros([3, 2], complex)
        r[1, 0] = 2 + 2j
        expected = r / l

        self._expectSparse(actual, expected)

    def testMulComplexTensorRealTensor(self):
        cxt = tc.Context()
        cxt.dense = tc.tensor.Dense.arange([3], 0, 3)
        cxt.sparse = tc.tensor.Sparse.zeros([2, 3], tc.C32)
        cxt.result = tc.after(cxt.sparse[0, 2].write(tc.C32(2 + 2j)), cxt.sparse * cxt.dense)

        actual = self.host.post(ENDPOINT, cxt)
        expected = np.zeros([2, 3], dtype=complex)
        expected[0, 2] = 2 + 2j
        expected = expected * np.arange(0, 3)

        self._expectSparse(actual, expected)

    def testMulComplexTensorComplexTensor(self):
        cxt = tc.Context()
        cxt.x1 = tc.tensor.Sparse.zeros([3], tc.C32)
        cxt.x2 = tc.tensor.Sparse.zeros([2, 3], tc.C32)
        cxt.result = tc.after((cxt.x1[1].write(1j), cxt.x2[0, 2].write(2+2j)), cxt.x1 * cxt.x2)

        actual = self.host.post(ENDPOINT, cxt)

        x1 = np.zeros([3], dtype=complex)
        x2 = np.zeros([2, 3], dtype=complex)
        x1[1] = 1j
        x2[0, 2] = 2+2j
        expected = x1 * x2

        self._expectSparse(actual, expected)

    def _expectSparse(self, actual, expected):
        (_schema, actual) = actual[str(tc.URI(tc.tensor.Sparse))]
        actual_np = np.zeros(expected.shape) + 0j
        for coord, n in actual:
            actual_np[tuple(coord)] = complex(*n)

        self.assertTrue(np.allclose(actual_np, expected))


def expect_number(as_json, dtype=tc.C64):
    return complex(*as_json[str(tc.URI(dtype))])


def load_dense(x, dtype):
    return tc.tensor.Dense.load(x.shape, x.flatten().tolist(), dtype)


if __name__ == "__main__":
    unittest.main()
