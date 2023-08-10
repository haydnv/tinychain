import itertools

import numpy as np
import tinychain as tc
import unittest

from .base import ClientTest

ENDPOINT = "/transact/hypothetical"


class ComplexNumberOpsTests(ClientTest):
    def testAngle(self):
        n = 3 + 4j

        cxt = tc.Context()
        cxt.n = n
        cxt.a = tc.C64(n).angle()
        cxt.b = tc.Complex(tc.URI("n")).angle()
        cxt.test = cxt.a == cxt.b

        self.assertTrue(self.host.post(ENDPOINT, cxt))

    def testConjugate(self):
        n = 2 - 2j

        cxt = tc.Context()
        cxt.z = tc.C64(n).conj()

        actual = self.host.post(ENDPOINT, cxt)
        self.assertEqual(expect_number(actual), n.conjugate())


def expect_number(as_json, dtype=tc.C64):
    return complex(*as_json[str(tc.URI(dtype))])


def load_dense(x, dtype=tc.C32):
    if issubclass(dtype, tc.Complex):
        data = [[n.real, n.imag] for n in x.flatten().tolist()]
    else:
        data = x.flatten().tolist()

    return tc.tensor.Dense.load(x.shape, data, dtype)


if __name__ == "__main__":
    unittest.main()
