import numpy as np
import tinychain as tc
import unittest

from testutils import ClientTest

ENDPOINT = "/transact/hypothetical"


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

    @unittest.skip
    def testSlogdet(self):
        x = np.arange(4)

        cxt = tc.Context()  # initialize an Op context
        cxt.x = tc.tensor.Dense.load(x.shape, tc.I32, x.flatten().tolist())  # load `x` as a TinyChain tensor
        cxt.slogdet = tc.linalg.slogdet(cxt.x)

        # the /transact/hypothetical endpoint will attempt to resolve whatever state you send it
        # in this case it will return the last state set in `cxt`
        actual_sign, actual_logdet = self.host.post(ENDPOINT, cxt)

        expected_sign, expected_logdet = np.linalg.slogdet(x)
        self.assertEqual(actual_sign, expected_sign)
        self.assertEqual(actual_logdet, expect_dense(expected_logdet, tc.F32))

    @unittest.skip
    def testSVD(self):
        m = 4
        n = 3
        matrix = np.arange(1, 1 + m * n).reshape(m, n)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((m, n), tc.F32, matrix.flatten().tolist())
        cxt.result = tc.linalg.svd(cxt.matrix)

        self.assertRaises(tc.error.NotImplemented, lambda: self.host.post(ENDPOINT, cxt))


def expect_dense(x, dtype):
    return {tc.uri(tc.tensor.Dense): [[list(x.shape), tc.uri(dtype)], x.flatten().tolist()]}


if __name__ == "__main__":
    unittest.main()
