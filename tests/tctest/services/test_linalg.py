import numpy as np
import time
import tinychain as tc
import unittest

from ..process import start_host

LIB_URI = tc.URI(tc.math.linalg.LinearAlgebra)
TENSOR_URI = str(tc.URI(tc.tensor.Dense))


class LinearAlgebraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = start_host(tc.math.NS)
        cls.host.put(tc.URI(tc.service.Library), LIB_URI[-3], LIB_URI[:-2])
        cls.host.install(tc.math.linalg.LinearAlgebra())

    def testQR_1(self):
        self._check_qr(4, 3, 1e-5)

    def testQR_2(self):
        self._check_qr(5, 5, 5e-4)

    def testQR_3(self):
        self._check_qr(5, 6, 5e-4)

    def _check_qr(self, m, n, threshold):
        matrix = np.random.random(n * m).reshape(n, m)
        tensor = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)

        q, r = self.host.post(LIB_URI.append("qr"), {'a': tensor})
        q, r = load_np(q), load_np(r)

        reconstruction = q @ r
        error = np.sum(abs(reconstruction - matrix))
        self.assertTrue(error < threshold)

    def testPLU(self):
        x = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 12.0, 6.0, 7.0], dtype=np.float32).reshape((3, 3))

        plu = self.host.post(LIB_URI.append("plu"), {'x': tc.tensor.Dense.load([3, 3], x.flatten().tolist(), tc.F32)})

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
        expected_sign, expected_logdet = np.linalg.slogdet(x)

        x = tc.tensor.Dense.load(x.shape, x.flatten().tolist(), tc.F32)
        actual_sign, actual_logdet = self.host.post(LIB_URI.append("slogdet"), tc.Map(x=x))

        actual_sign = load_np(actual_sign)
        actual_logdet = load_np(actual_logdet)

        self.assertTrue((actual_sign == expected_sign).all())
        self.assertTrue((abs((actual_logdet - expected_logdet)) < 1e-4).all())

    def testSVD_1x1(self):
        n = 1
        m = 1
        matrix = np.random.random(n * m).reshape(n, m)
        tensor = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)

        start = time.time()
        result = self.host.post(LIB_URI.append("svd"), tc.Map(A=tensor, l=n, epsilon=tc.F32(1e-7), max_iter=30))
        elapsed = time.time() - start

        print(f"{n}x{m} SVD ran in {elapsed}s")

        U, s, V = result
        actual = (load_np(U), load_np(s), load_np(V))
        self._check_svd(matrix, actual)

    def testMatrixSVD_NltM(self):
        n = 4
        m = 5
        matrix = np.random.random(n * m).reshape(n, m)
        tensor = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)

        start = time.time()
        result = self.host.post(LIB_URI.append("svd"), tc.Map(A=tensor, l=n, epsilon=tc.F32(1e-7), max_iter=30))
        elapsed = time.time() - start

        print(f"{n}x{m} SVD ran in {elapsed}s")

        U, s, V = result
        actual = (load_np(U), load_np(s), load_np(V))
        self._check_svd(matrix, actual)

    def testMatrixSVD_NgtM(self):
        n = 4
        m = 3
        matrix = np.random.random(n * m).reshape(n, m)
        tensor = tc.tensor.Dense.load((n, m), matrix.flatten().tolist(), tc.F32)

        start = time.time()
        result = self.host.post(LIB_URI.append("svd"), tc.Map(A=tensor, l=n, epsilon=tc.F32(1e-7), max_iter=200))
        elapsed = time.time() - start

        print(f"{n}x{m} SVD ran in {elapsed}s")

        U, s, V = result
        actual = (load_np(U), load_np(s), load_np(V))
        self._check_svd(matrix, actual)

    def testParallelSVD_NltM(self):
        num_matrices = 25
        n = 2
        m = 3
        shape = [num_matrices, n, m]
        matrices = np.random.random(np.product(shape)).reshape(shape)
        tensor = tc.tensor.Dense.load(shape, matrices.flatten().tolist(), tc.F32)

        start = time.time()
        result = self.host.post(LIB_URI.append("svd"), tc.Map(A=tensor, l=n, epsilon=1e-7, max_iter=10))
        elapsed = time.time() - start

        print(f"{num_matrices}x{n}x{m} SVD ran in {elapsed}s")

        U, s, V = result
        U = load_np(U)
        s = load_np(s)
        V = load_np(V)

        for i in range(num_matrices):
            expected = matrices[i]
            actual = (U[i], s[i], V[i])
            self._check_svd(expected, actual)

    def testParallelSVD_NgtM(self):
        num_matrices = 10
        n = 3
        m = 2
        shape = [num_matrices, n, m]
        matrices = np.random.random(np.product(shape)).reshape(shape)
        tensor = tc.tensor.Dense.load(shape, matrices.flatten().tolist(), tc.F32)

        start = time.time()
        result = self.host.post(
            LIB_URI.append("svd"),
            tc.Map(A=tensor, l=n, epsilon=1e-7, max_iter=30))

        elapsed = time.time() - start

        print(f"{num_matrices}x{n}x{m} SVD ran in {elapsed}s")

        U, s, V = result
        U = load_np(U)
        s = load_np(s)
        V = load_np(V)

        for i in range(num_matrices):
            expected = matrices[i]
            actual = (U[i], s[i], V[i])
            self._check_svd(expected, actual)

    def _check_svd(self, expected, actual, threshold=5e-4):
        (U, s, V) = actual
        actual = (U @ (np.eye(s.shape[0], s.shape[0]) * s) @ V)
        self.assertTrue(abs(actual - expected).sum() < threshold)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.host.stop()


def load_np(as_json, dtype=float):
    shape = as_json[TENSOR_URI][0][0]
    return np.array(as_json[TENSOR_URI][1], dtype).reshape(shape)
