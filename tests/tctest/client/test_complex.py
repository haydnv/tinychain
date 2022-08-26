import tinychain as tc
import unittest

import numpy as np

from .base import ClientTest

ENDPOINT = "/transact/hypothetical"
C64_URI = str(tc.URI(tc.C64))


# TODO: fix failing tests
# TODO: move to the host test package
class ComplexNumberOpsTests(ClientTest):
    @unittest.skip
    def testReal(self):
        n = complex(1, 2)

        cxt = tc.Context()
        cxt.z = tc.C64(n).real()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, n.real)

    @unittest.skip
    def testImag(self):
        n = complex(1, 2)

        cxt = tc.Context()
        cxt.z = tc.C64(n).imag()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, n.imag)

    @unittest.skip
    def testConjugate(self):
        n = complex(2, -2)

        cxt = tc.Context()
        cxt.z = tc.C64(n).conj()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(parse(actual), n.conjugate())

    @unittest.skip
    def testAbs(self):
        n = complex(3, 4)

        cxt = tc.Context()
        cxt.z = tc.C64(n).abs()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, abs(n))

    @unittest.skip
    def testAngle(self):
        n = 3 + 4j

        cxt = tc.Context()
        cxt.z = tc.C64(n).angle()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(actual, np.angle(n))

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

    @unittest.skip
    def testExp(self):
        n = 0.25j

        cxt = tc.Context()
        cxt.z = (tc.C64(n) * np.pi).exp()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertAlmostEqual(actual[C64_URI], np.exp(n * np.pi))


def parse(as_json):
    return complex(*as_json[C64_URI])


if __name__ == "__main__":
    unittest.main()
