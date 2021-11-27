import numpy as np
import unittest
import testutils
import tinychain as tc


def truncated_normal(size, mean=0., std=None):
    std = std if std else size**0.5

    while True:
        dist = np.random.normal(mean, std, size)
        truncate = np.abs(dist) > mean + (std * 2)
        if truncate.any():
            new_dist = np.random.normal(mean, std, size) * truncate
            dist *= np.logical_not(truncate)
            dist += new_dist
        else:
            return dist


def create_layer(input_size, output_size, activation):
    shape = (input_size, output_size)
    bias = tc.tensor.Dense.load([output_size], tc.F32, truncated_normal(output_size).tolist())
    weights = tc.tensor.Dense.load(shape, tc.F32, truncated_normal(input_size * output_size).tolist())
    return tc.ml.dnn.layer(weights, bias, activation)


class App(tc.Cluster):
    __uri__ = tc.URI("/test/app")

    # def _configure(self):
    #     layers = [create_layer(2, 2, tc.ml.Sigmoid), create_layer(2, 1, tc.ml.ReLU)]
    #     self.neural_net = tc.ml.dnn.neural_net(layers)

    @tc.get_method
    def up(self) -> tc.Bool:
        return True


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_app", [App], wait_time=2.)

    def testApp(self):
        self.assertTrue(self.host.get("/test/app/up"))



if __name__ == "__main__":
    unittest.main()
