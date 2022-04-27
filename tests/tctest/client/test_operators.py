import typing as t

import numpy as np
import tinychain as tc
import unittest

from .base import ClientTest

ENDPOINT = "/transact/hypothetical"
TENSOR_URI = str(tc.uri(tc.tensor.Dense))


class OperatorTests(ClientTest):
    def testDiagonal(self):
        x = tc.ml.Variable.arange([3, 3], 0, 9)
        x_slice = x[0:2]
        loss = tc.tensor.Dense.ones([2, 3])
        grads = tc.math.operator.operator(x_slice).gradients(loss)

        cxt = tc.Context()
        cxt.grads = grads  # TODO: support referencing an Operator from a Context

        response = self.host.post(ENDPOINT, cxt)
        grad = response[tc.util.hex_id(x)]
        self.assertEqual(load_np(grad).shape, (3, 3))


def expect_dense(x, dtype):
    return {tc.uri(tc.tensor.Dense): [[list(x.shape), tc.uri(dtype)], x.flatten().tolist()]}


def load_np(as_json: t.Dict[str, t.Any], dtype=np.float32) -> np.ndarray:
    shape = as_json[TENSOR_URI][0][0]
    return np.array(as_json[TENSOR_URI][1], dtype).reshape(shape)


if __name__ == "__main__":
    unittest.main()
