import unittest
import tinychain as tc

from testutils import start_host


CONSERVED = tc.Number(20)


class Balance(tc.Cluster):
    __uri__ = tc.URI("/app/balance")

    def _configure(self):
        self.weight = tc.chain.Sync(tc.UInt(10))


class Left(Balance):
    __uri__ = "http://127.0.0.1:8702" + tc.uri(Balance) + "/left"

    @tc.put_method
    def weigh(self, txn, key: tc.Nil, new_value: tc.Number):
        right = tc.use(Right)

        txn.total = CONSERVED
        txn.update = tc.After(
            self.weight.set(new_value),
            right.weigh(None, txn.total - new_value))

        return tc.If(self.weight == new_value, None, txn.update)


class Right(Balance):
    __uri__ = "http://127.0.0.1:8703" + tc.uri(Balance) + "/right"

    @tc.put_method
    def weigh(self, txn, key: tc.Nil, new_value: tc.Number):
        left = tc.use(Left)

        txn.total = CONSERVED
        txn.update = tc.After(
            self.weight.set(new_value),
            left.weigh(None, txn.total - new_value))

        return tc.If(self.weight == new_value, None, txn.update)


class InteractionTests(unittest.TestCase):
    def testUpdate(self):
        left = start_host("test_multi_host_left", [Left])
        right = start_host("test_multi_host_right", [Right])

        left.put("/app/balance/left/weigh", None, 5)
        print_lines(5)
        self.assertEqual(left.get("/app/balance/left/weight"), 5)
        self.assertEqual(right.get("/app/balance/right/weight"), 15)


def print_lines(n):
    for _ in range(n):
        print()


if __name__ == "__main__":
    unittest.main()

