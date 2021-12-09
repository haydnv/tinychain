import numpy as np
import unittest
import testutils
import tinychain as tc

LEARNING_RATE = 0.01
SHAPE = [(2, 2), (2, 1)]


class App(tc.Cluster):
    __uri__ = tc.URI("/test/app")

    def _configure(self):
        dnn = tc.ml.dnn.DNN.create(SHAPE)
        self.net = tc.chain.Sync(dnn)

    @tc.get_method
    def up(self) -> tc.Bool:
        return True

    @tc.post_method
    def reset(self, new_layers: tc.Tuple):
        return self.net.write(new_layers)

    @tc.post_method
    def train(self, inputs: tc.tensor.Dense, labels: tc.tensor.Dense):
        output = self.net.train(inputs, lambda output: ((output - labels) ** 2) * LEARNING_RATE)
        return (labels - output).abs().sum()


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = testutils.start_host("test_app", [App], wait_time=2.)

    def testApp(self):
        self.assertTrue(self.host.get("/test/app/up"))

    def testTrain(self):
        tc.write_cluster(App, "test_app.json", overwrite=True)

        new_layers = []
        for i, o in SHAPE:
            weights = tc.tensor.Dense.load([i, o], tc.F32, np.random.random(i * o).tolist())
            bias = tc.tensor.Dense.load([o], tc.F32, np.random.random(o).tolist())
            new_layers.append((weights, bias))

        inputs = tc.tensor.Dense.load([4, 2], tc.Bool, [
            True, True,
            True, False,
            False, True,
            False, False,
        ])

        labels = tc.tensor.Dense.load([4, 1], tc.Bool, [
            False,
            True,
            True,
            False,
        ])

        self.host.post("/test/app/reset", {"new_layers": new_layers})

        i = 0
        while True:
            error = self.host.post("/test/app/train", {"inputs": inputs, "labels": labels})
            print(f"error is {error} at iteration {i}")

            if error < 0.5:
                print(f"converged in {i} iterations")
                return
            else:
                i += 1


if __name__ == "__main__":
    unittest.main()
