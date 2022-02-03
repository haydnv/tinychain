import typing as t

import numpy as np
import tinychain as tc
import unittest

from testutils import ClientTest

ENDPOINT = "/transact/hypothetical"
TENSOR_URI = str(tc.uri(tc.tensor.Dense))

def _load_dense_tensor_from_json_to_numpy(jsn: t.Dict[str, t.Any]) -> np.ndarray:
    return np.array(jsn[TENSOR_URI][1], dtype=np.float32)\
        .reshape(jsn[TENSOR_URI][0][0])


class LinearAlgebraTests(ClientTest):
    def testDiagonal(self):
        x = np.arange(0, 9).reshape(3, 3)

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, tc.I32, x.flatten().tolist())
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
        cxt.x = tc.tensor.Dense.load(x.shape, tc.I32, x.flatten().tolist())
        cxt.diag = tc.tensor.Dense.load([size], tc.I32, diag)
        cxt.result = tc.After(tc.linalg.set_diagonal(cxt.x, cxt.diag), cxt.x)

        x[range(size), range(size)] = diag
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, expect_dense(x, tc.I32))

    def testMatmul(self):
        l = np.random.random([2, 3, 4])
        r = np.random.random([2, 4, 5])

        cxt = tc.Context()
        cxt.l = tc.tensor.Dense.load(l.shape, tc.F32, l.flatten().tolist())
        cxt.r = tc.tensor.Dense.load(r.shape, tc.F32, r.flatten().tolist())
        cxt.result = tc.linalg.matmul(cxt.l, cxt.r)

        expected = np.matmul(l, r)
        actual = self.host.post(ENDPOINT, cxt)
        actual = actual[tc.uri(tc.tensor.Dense)][1]

        self.assertTrue(np.allclose(expected.flatten(), actual))

    def testNorm(self):
        shape = [2, 3, 4]
        matrices = np.arange(24).reshape(shape)

        cxt = tc.Context()
        cxt.matrices = tc.tensor.Dense.load(shape, tc.I32, matrices.flatten().tolist())
        cxt.result = tc.linalg.norm(tensor=cxt.matrices)

        expected = [np.linalg.norm(matrix) for matrix in matrices]

        actual = self.host.post(ENDPOINT, cxt)
        actual = actual[tc.uri(tc.tensor.Dense)][1]

        self.assertEqual(actual, expected)

    def testQR(self):
        THRESHOLD = 1.

        m = 4
        n = 3
        matrix = np.arange(1, 1 + m * n).reshape(m, n)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((m, n), tc.F32, matrix.flatten().tolist())
        cxt.qr = tc.linalg.qr
        cxt.result = cxt.qr(x=cxt.matrix)
        cxt.reconstruction = tc.tensor.einsum("ij,jk->ik", cxt.result)
        cxt.threshold = ((cxt.reconstruction - cxt.matrix) < THRESHOLD).all()

        response = self.host.post(ENDPOINT, cxt)
        self.assertTrue(response)
    
    def testPLU(self):
        x = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 12.0, 6.0, 7.0], dtype=np.float32).reshape((3, 3))

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load([3, 3], tc.F32, x.flatten().tolist())
        cxt.plu = tc.linalg.plu
        cxt.plu_factorization = cxt.plu(x=cxt.x)
        plu = self.host.post(ENDPOINT, cxt)

        p = _load_dense_tensor_from_json_to_numpy(plu['p'])
        l = _load_dense_tensor_from_json_to_numpy(plu['l'])
        u = _load_dense_tensor_from_json_to_numpy(plu['u'])

        self.assertTrue((p @ l @ u == x).all())
        self.assertTrue([
            np.allclose(l, np.tril(l)),
            np.allclose(u, np.triu(u)),
            all([np.sum(p[i, :]) == np.sum(p[:, i]) == 1.0 for i in range(p.shape[0])]),
        ])

    def testSlogdet(self):
        x = np.random.randint(-100, 100, 64).reshape(4, 4, 4)

        cxt = tc.Context()
        cxt.x = tc.tensor.Dense.load(x.shape, tc.F32, x.flatten().tolist())
        cxt.slogdet = tc.linalg.slogdet
        cxt.result_slogdet = cxt.slogdet(x=cxt.x)

        actual_sign, actual_logdet = self.host.post(ENDPOINT, cxt)
        actual_sign = _load_dense_tensor_from_json_to_numpy(actual_sign)
        actual_logdet = _load_dense_tensor_from_json_to_numpy(actual_logdet)
        expected_sign, expected_logdet = np.linalg.slogdet(x)
        self.assertTrue((actual_sign == expected_sign).all())
        self.assertTrue((abs((actual_logdet - expected_logdet)) < 1e-4).all())

    def testSVD(self):
        m = 4
        n = 3
        matrix = np.arange(1, 1 + m * n).reshape(m, n)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((m, n), tc.F32, matrix.flatten().tolist())
        cxt.result = tc.linalg.svd(cxt.matrix)
        cxt.U_shape = cxt.result['U'].shape
        response = self.host.post(ENDPOINT, cxt)
        self.assertEqual(response, [m, n])
        cxt.W_shape = cxt.result['W'].shape
        response = self.host.post(ENDPOINT, cxt)
        self.assertEqual(response, [n, n])
        cxt.V_shape = cxt.result['V'].shape
        response = self.host.post(ENDPOINT, cxt)
        self.assertEqual(response, [n, n])


def expect_dense(x, dtype):
    return {tc.uri(tc.tensor.Dense): [[list(x.shape), tc.uri(dtype)], x.flatten().tolist()]}


if __name__ == "__main__":
    unittest.main()
