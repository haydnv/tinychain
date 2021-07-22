import unittest
import tinychain as tc

from testutils import start_host


CONSERVED = tc.Number(20)


class Balance(tc.Cluster):
    __uri__ = tc.URI("/app/balance")

    def _configure(self):
        self.weight = tc.chain.Sync(tc.UInt(10))


class Left(Balance):
    __uri__ = tc.uri(Balance) + "/left"

    @tc.post_method
    def weigh(self, txn, weight: tc.Number):
        right = tc.use(Right)

        txn.total = CONSERVED
        txn.update = tc.After(
            self.weight.set(weight),
            right.weigh({"weight": (txn.total - weight)}))

        return tc.If(self.weight == weight, None, txn.update)


class Right(Balance):
    __uri__ = tc.uri(Balance) + "/right"

    @tc.post_method
    def weigh(self, txn, weight: tc.Number):
        left = tc.use(Left)

        txn.total = CONSERVED
        txn.update = tc.After(
            self.weight.set(weight),
            left.weigh({"weight": (txn.total - weight)}))

        return tc.If(self.weight == weight, None, txn.update)


class InteractionTests(unittest.TestCase):
    def testStartup(self):
        expected = 10

        host = start_host("test_multi_cluster_startup", [Left, Right])

        actual = host.get("/app/balance/left/weight")
        self.assertEqual(expected, actual)

        actual = host.get("/app/balance/right/weight")
        self.assertEqual(expected, actual)

    def testUpdate(self):
        host = start_host("test_multi_cluster_update", [Left, Right])
        host.post("/app/balance/right/weigh", {"weight": 5})
        self.assertEqual(host.get("/app/balance/right/weight"), 5)
        self.assertEqual(host.get("/app/balance/left/weight"), 15)


if __name__ == "__main__":
    unittest.main()
