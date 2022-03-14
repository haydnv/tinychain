import typing as t

import numpy as np
import tinychain as tc
import unittest

from testutils import ClientTest

ENDPOINT = "/transact/hypothetical"
TENSOR_URI = str(tc.uri(tc.tensor.Dense))


class LinearAlgebraTests(ClientTest):
    def testDiagonal(self):
        x = np.arange(0, 9).reshape(3, 3)

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.I32)
        cxt.diag = tc.linalg.diagonal(cxt.x)

        expected = np.diag(x)
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(expected, tc.I32))

    def testSetDiagonal(self):
        size = 3
        shape = [size, size]
        x = np.arange(0, size**2).reshape(shape)
        diag = [2] * size

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.I32)
        cxt.diag = tc.tensor.Dense.load([size], diag, tc.I32)
        cxt.result = tc.After(tc.linalg.set_diagonal(cxt.x, cxt.diag), cxt.x)

        x[range(size), range(size)] = diag
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(x, tc.I32))

    def testMatmul(self):
        l = np.random.random([2, 3, 4])
        r = np.random.random([2, 4, 5])

        cxt = tc.Context()
        cxt.l = tc.tensor.Dense.load(l.shape, l.flatten().tolist(), tc.F32)
        cxt.r = tc.tensor.Dense.load(r.shape, r.flatten().tolist(), tc.F32)
        cxt.result = cxt.l @ cxt.r

        expected = np.matmul(l, r)
        actual = self.host.post(ENDPOINT, cxt)
        actual = actual[tc.uri(tc.tensor.Dense)][1]

        self.assertTrue(np.allclose(expected.flatten(), actual))

    def testNorm(self):
        shape = [2, 3, 4]
        matrices = np.arange(24).reshape(shape)

        cxt = tc.Context()
        cxt.matrices = tc.tensor.Dense.load(shape, matrices.flatten().tolist(), tc.I32)
        cxt.result = tc.linalg.norm(tensor=cxt.matrices)

        expected = [np.linalg.norm(matrix) for matrix in matrices]

        actual = self.host.post(ENDPOINT, cxt)
        actual = actual[tc.uri(tc.tensor.Dense)][1]

        self.assertEqual(actual, expected)

    def testQR_1(self):
        THRESHOLD = 1e-6

        n = 3
        m = 4
        matrix = np.random.random(n * m).reshape(n, m)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)
        cxt.qr = tc.linalg.qr
        cxt.q, cxt.r = cxt.qr(a=cxt.matrix).unpack(2)
        cxt.reconstruction = tc.tensor.einsum("ij,jk->ik", [cxt.q, cxt.r])
        cxt.threshold = ((cxt.reconstruction - cxt.matrix).abs().sum() < THRESHOLD)
        cxt.error = (cxt.reconstruction - cxt.matrix)
        response = self.host.post(ENDPOINT, cxt)

        self.assertTrue(response)

    def testQR_2(self):
        THRESHOLD = 5e-4

        n = 5
        m = 5
        matrix = np.random.random(n * m).reshape(n, m)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)
        cxt.qr = tc.linalg.qr
        cxt.q, cxt.r = cxt.qr(a=cxt.matrix).unpack(2)
        cxt.reconstruction = tc.tensor.einsum("ij,jk->ik", [cxt.q, cxt.r])
        cxt.threshold = ((cxt.reconstruction - cxt.matrix).abs().sum() < THRESHOLD)
        response = self.host.post(ENDPOINT, cxt)

        self.assertTrue(response)

    def testQR_3(self):
        THRESHOLD = 5e-4

        n = 6
        m = 5
        matrix = np.random.random(n * m).reshape(n, m)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)
        cxt.qr = tc.linalg.qr
        cxt.q, cxt.r = cxt.qr(a=cxt.matrix).unpack(2)
        cxt.reconstruction = tc.tensor.einsum("ij,jk->ik", [cxt.q, cxt.r])
        cxt.error = ((cxt.reconstruction - cxt.matrix).abs().sum() < THRESHOLD)
        response = self.host.post(ENDPOINT, cxt)

        self.assertTrue(response)

    def testPLU(self):
        x = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 12.0, 6.0, 7.0], dtype=np.float32).reshape((3, 3))

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load([3, 3], x.flatten().tolist(), tc.F32)
        cxt.plu = tc.linalg.plu
        cxt.plu_factorization = cxt.plu(x=cxt.x)
        plu = self.host.post(ENDPOINT, cxt)

        p = load_np(plu['p'])
        l = load_np(plu['l'])
        u = load_np(plu['u'])

        self.assertTrue((p @ l @ u == x).all())
        self.assertTrue([
            np.allclose(l, np.tril(l)),
            np.allclose(u, np.triu(u)),
            all([np.sum(p[i, :]) == np.sum(p[:, i]) == 1.0 for i in range(p.shape[0])]),
        ])

    def testSlogdet(self):
        x = np.random.randint(-100, 100, 64).reshape(4, 4, 4)

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.F32)
        cxt.slogdet = tc.linalg.slogdet
        cxt.result_slogdet = cxt.slogdet(x=cxt.x)

        actual_sign, actual_logdet = self.host.post(ENDPOINT, cxt)
        actual_sign = load_np(actual_sign)
        actual_logdet = load_np(actual_logdet)
        expected_sign, expected_logdet = np.linalg.slogdet(x)
        self.assertTrue((actual_sign == expected_sign).all())
        self.assertTrue((abs((actual_logdet - expected_logdet)) < 1e-4).all())


class SVDTests(ClientTest):
    def testSvd_1x1(self):
        n = 1
        m = 1
        matrix = np.random.random(n * m).reshape(n, m)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)
        cxt.svd = tc.linalg.svd
        cxt.result = cxt.svd(A=cxt.matrix, l=n, epsilon=tc.F32(1e-7), max_iter=30)

        result = self.host.post(ENDPOINT, cxt)
        U, s, V = result
        actual = (load_np(U), load_np(s), load_np(V))
        self._check(matrix, actual)

    def testMatrixSvd_NltM(self):
        n = 4
        m = 5
        matrix = np.random.random(n * m).reshape(n, m)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)
        cxt.svd = tc.linalg.svd
        cxt.result = cxt.svd(A=cxt.matrix, l=n, epsilon=tc.F32(1e-7), max_iter=30)

        result = self.host.post(ENDPOINT, cxt)
        U, s, V = result
        actual = (load_np(U), load_np(s), load_np(V))
        self._check(matrix, actual)

    def testMatrixSVD_NgtM(self):
        n = 4
        m = 3
        matrix = np.random.random(n * m).reshape(n, m)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)
        cxt.svd = tc.linalg.svd
        cxt.result = cxt.svd(A=cxt.matrix, l=n, epsilon=tc.F32(1e-7), max_iter=200)

        result = self.host.post(ENDPOINT, cxt)
        U, s, V = result
        actual = (load_np(U), load_np(s), load_np(V))
        self._check(matrix, actual)

    def testParallelSVD_NltM(self):
        num_matrices = 1
        n = 2
        m = 3
        shape = [num_matrices, n, m]
        matrices = np.random.random(np.product(shape)).reshape(shape)

        cxt = tc.Context()
        cxt.matrices = tc.tensor.Dense.load(shape, matrices.flatten().tolist(), tc.F32)
        cxt.svd = tc.linalg.svd
        cxt.result = cxt.svd(A=cxt.matrices, l=n, epsilon=1e-7, max_iter=10)

        result = self.host.post(ENDPOINT, cxt)

        U, s, V = result
        U = load_np(U)
        s = load_np(s)
        V = load_np(V)

        for i in range(num_matrices):
            expected = matrices[i]
            actual = (U[i], s[i], V[i])
            self._check(expected, actual)

    def testParallelSVD_NgtM(self):
        num_matrices = 1
        n = 3
        m = 2
        shape = [num_matrices, n, m]
        matrices = np.random.random(np.product(shape)).reshape(shape)

        cxt = tc.Context()
        cxt.matrices = tc.tensor.Dense.load(shape, matrices.flatten().tolist(), tc.F32)
        cxt.svd = tc.linalg.svd
        cxt.result = cxt.svd(A=cxt.matrices, l=n, epsilon=1e-7, max_iter=30)

        result = self.host.post(ENDPOINT, cxt)

        U, s, V = result
        U = load_np(U)
        s = load_np(s)
        V = load_np(V)

        for i in range(num_matrices):
            expected = matrices[i]
            actual = (U[i], s[i], V[i])
            self._check(expected, actual)

    def _check(self, expected, actual, threshold=5e-4):
        (U, s, V) = actual

        actual = (U @ (np.eye(s.shape[0], s.shape[0]) * s) @ V)
        self.assertTrue(abs(actual - expected).sum() < threshold)


def expect_dense(x, dtype):
    return {tc.uri(tc.tensor.Dense): [[list(x.shape), tc.uri(dtype)], x.flatten().tolist()]}


def load_np(as_json: t.Dict[str, t.Any], dtype=np.float32) -> np.ndarray:
    shape = as_json[TENSOR_URI][0][0]
    return np.array(as_json[TENSOR_URI][1], dtype).reshape(shape)


if __name__ == "__main__":
    unittest.main()
