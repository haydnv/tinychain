import math
import numpy as np
import unittest
import testutils
import tinychain as tc

LAYER_CONFIG = [(2, 2, tc.ml.ReLU()), (2, 1, tc.ml.Sigmoid())]
LEARNING_RATE = 0.1
BATCH_SIZE = 25


class App(tc.Cluster):
    __uri__ = tc.URI("/test/app")

    def _configure(self):
        dnn = tc.ml.dnn.DNN.create(LAYER_CONFIG)
        self.net = tc.chain.Sync(dnn)

    @tc.get_method
    def up(self) -> tc.Bool:
        return True

    @tc.post_method
    def reset(self, new_layers: tc.Tuple):
        return self.net.write(new_layers)

    @tc.post_method
    def train(self, inputs: tc.tensor.Dense, labels: tc.tensor.Dense):
        # TODO: implement this using a background task
        pass


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_app", [App])

    def testApp(self):
        self.assertTrue(self.host.get("/test/app/up"))

    def testTrain(self):
        np.random.seed()

        new_layers = []
        for i, o, _ in LAYER_CONFIG:
            weights = load(truncated_normal(i * o).reshape([i, o]))
            bias = load(truncated_normal(o))
            new_layers.append((weights, bias))

        self.host.post("/test/app/reset", {"new_layers": new_layers})

        # TODO


def truncated_normal(size, mean=0., std=None):
    std = std if std else math.sqrt(size)

    while True:
        dist = np.random.normal(mean, std, size)
        truncate = np.abs(dist) > mean + (std * 2)
        if truncate.any():
            new_dist = np.random.normal(mean, std, size) * truncate
            dist *= np.logical_not(truncate)
            dist += new_dist
        else:
            return dist


def load(nparray, dtype=tc.F64):
    return tc.tensor.Dense.load(nparray.shape, dtype, nparray.flatten().tolist())


if __name__ == "__main__":
    unittest.main()
