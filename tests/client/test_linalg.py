import numpy as np
import tinychain as tc
import unittest

from testutils import ClientTest

ENDPOINT = "/transact/hypothetical"


class LinearAlgebraTests(ClientTest):
    def testBidiagonalization(self):
        m = 5
        n = 3

        # TODO: figure out why this is so inaccurate
        threshold = m * n * 2

        matrix = np.arange(m * n).reshape(m, n)

        cxt = tc.Context()
        cxt.bidiagonalize = tc.linalg.bidiagonalize
        cxt.matrix = tc.tensor.Dense.load([m, n], tc.F64, matrix.flatten().tolist())
        cxt.result = cxt.bidiagonalize(x=cxt.matrix)

        cxt.reconstruction = tc.tensor.einsum("ij,jk,kl->ik", [
            tc.tensor.Dense(cxt.result["U"]),
            cxt.result["A"],
            tc.tensor.Dense(cxt.result["V"])
        ])

        cxt.threshold = ((cxt.matrix - cxt.reconstruction) < threshold).all()

        response = self.host.post(ENDPOINT, cxt)
        self.assertTrue(response)

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

    def testSVD(self):
        m = 4
        n = 3
        matrix = np.arange(1, 1 + m * n).reshape(m, n)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((m, n), tc.F32, matrix.flatten().tolist())
        cxt.svd = tc.linalg.svd
        cxt.result = cxt.svd(x=cxt.matrix)

        response = self.host.post(ENDPOINT, cxt)
        print(response)


if __name__ == "__main__":
    unittest.main()
