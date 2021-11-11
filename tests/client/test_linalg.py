import numpy as np
import tinychain as tc
import unittest

from testutils import ClientTest

ENDPOINT = "/transact/hypothetical"


class LinearAlgebraTests(ClientTest):
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
        THRESHOLD = 0.01

        m = 3
        n = 3
        matrix = np.arange(1, 1 + m * n).reshape(m, n)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load((m, n), tc.F32, matrix.flatten().tolist())
        cxt.qr = tc.linalg.qr
        cxt.result = cxt.qr(matrix=cxt.matrix)
        cxt.test = ((tc.tensor.einsum("ij,jk->ik", cxt.result) - cxt.matrix) < THRESHOLD).all()

        cxt = tc.to_json(cxt)
        self.assertTrue(self.host.post(ENDPOINT, cxt))


if __name__ == "__main__":
    unittest.main()
