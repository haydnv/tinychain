import tinychain as tc
import unittest

import numpy as np

from base import ClientTest

ENDPOINT = "/transact/hypothetical"
TENSOR_URI = str(tc.URI(tc.tensor.Dense))
key = '/state/scalar/value/number/complex/64'

class ComplexNumberOpsTests(ClientTest):
    def testReal(self):
        cxt = tc.Context()
        cxt.z = tc.C64((1, 2)).real()
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], 1)

    def testImag(self):
        cxt = tc.Context()
        cxt.z = tc.C64((1, 2)).imag()
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], 2)

    def testConjugate(self):
        cxt = tc.Context()
        cxt.z = tc.C64((2, 2)).conj()
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], [2, -2])

    def testAbs(self):
        cxt = tc.Context()
        cxt.z = tc.C64((3, 4)).abs()
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], 5)

    def testArg(self):
        cxt = tc.Context()
        cxt.z = tc.C64((3, 4)).arg()
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], np.angle(3 + 4j))

    def testSum(self):
        cxt = tc.Context()
        cxt.z = tc.C64((4, 4)) + tc.C64((4, 4))
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], [8, 8])

    def testSub(self):
        cxt = tc.Context()
        cxt.z = tc.C64((4, 4)) - tc.C64((4, 4))
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], 0)

    def testProdComplex(self):
        cxt = tc.Context()
        cxt.z = tc.C64((1, 2)) * tc.C64((3, 4))
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], [5, 10])

    def testProdReal(self):
        cxt = tc.Context()
        cxt.z = tc.C64((1, 2)) * tc.I16(2)
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], [2, 4])

    def testDivComplex(self):
        cxt = tc.Context()
        cxt.z = tc.C64((4, 4)) / tc.C64((2, 2))
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], 2)

    def testDivReal1(self):
        cxt = tc.Context()
        cxt.z = tc.C64((4, 4)) / tc.I16(2)
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], [2, 2])

    def testDivReal2(self):
        cxt = tc.Context()
        cxt.z = tc.I16(2) / tc.C64((2, 2)) 
        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual[key], [0.5, 0.5])

    def testPow(self):
        cxt = tc.Context()
        cxt.z = tc.C64((2.3, 3.4)).pow(3)
        actual = self.host.post(ENDPOINT, cxt)
        expected = complex(2.3, 3.4)**3
        self.assertAlmostEqual(actual[key], expected)

    def testExp(self):
        cxt = tc.Context()
        cxt.pi = 3.14159265359 
        cxt.z = (tc.C64((0, 0.25)) * cxt.pi).exp()
        actual = self.host.post(ENDPOINT, cxt)
        expected = np.exp(0.25j * np.pi)
        self.assertAlmostEqual(actual[key], expected)


if __name__ == "__main__":
    unittest.main()
