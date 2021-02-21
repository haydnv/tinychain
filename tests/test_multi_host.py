import unittest
import tinychain as tc

from testutils import start_host


CONSERVED = tc.Number(20)


class Balance(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("/app/balance")

    def configure(self):
        self.weight = tc.Chain.Sync(10)


class Left(Balance):
    __uri__ = "http://127.0.0.1:8702" + tc.uri(Balance)

    @tc.put_method
    def weigh(self, txn, key: tc.Nil, new_value: tc.Number):
        right = tc.use(Right)

        txn.total = CONSERVED
        txn.current = self.weight.subject()
        txn.update = tc.After(
            self.weight.set(new_value),
            right.weigh(None, txn.total - new_value))

        return tc.If(txn.current == new_value, None, txn.update)


class Right(Balance):
    __uri__ = "http://127.0.0.1:8703" + tc.uri(Balance)

    @tc.put_method
    def weigh(self, txn, key: tc.Nil, new_value: tc.Number):
        left = tc.use(Left)

        txn.total = CONSERVED
        txn.current = self.weight.subject()
        txn.update = tc.After(
            self.weight.set(new_value),
            left.weigh(None, txn.total - new_value))

        return tc.If(txn.current == new_value, None, txn.update)


class InteractionTests(unittest.TestCase):
    def testUpdate(self):
        print(tc.uri(Right).port())
        assert tc.uri(Right).port() == 8703
        left = start_host("test_interaction_left", [Left])
        right = start_host("test_interaction_right", [Right])

        left.put("/app/balance/weigh", None, 5)
        print_lines(5)
        self.assertEqual(left.get("/app/balance/weight"), 5)
        self.assertEqual(right.get("/app/balance/weight"), 15)


def print_lines(n):
    for _ in range(n):
        print()


if __name__ == "__main__":
    unittest.main()

