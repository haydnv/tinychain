import unittest
import tinychain as tc

from testutils import start_host


CONSERVED = tc.Number(20)


class Balance(tc.Cluster, metaclass=tc.Meta):
    __uri__ = tc.URI("/app/balance")

    def configure(self):
        self.weight = tc.Chain.Sync(10)


class Left(Balance):
    __uri__ = tc.uri(Balance) + "/left"

    @tc.put_method
    def weigh(self, txn, key: tc.Nil, new_value: tc.Number):
        right = tc.use(Right)

        txn.total = CONSERVED
        txn.current = self.weight.subject()
        txn.update = tc.After(
            right.weigh(None, txn.total - new_value),
            self.weight.set(new_value))

        return tc.If(txn.current == new_value, None, txn.update)


class Right(Balance):
    __uri__ = tc.uri(Balance) + "/right"

    @tc.put_method
    def weigh(self, txn, key: tc.Nil, new_value: tc.Number):
        left = tc.use(Left)

        txn.total = CONSERVED
        txn.current = self.weight.subject()
        txn.update = tc.After(
            left.weigh(None, txn.total - new_value),
            self.weight.set(new_value))

        return tc.If(txn.current == new_value, None, txn.update)


class InteractionTests(unittest.TestCase):
    def testStartup(self):
        expected = 10

        host = start_host("test_interaction", [Left, Right])

        actual = host.get("/app/balance/left/weight")
        self.assertEqual(expected, actual)

        actual = host.get("/app/balance/right/weight")
        self.assertEqual(expected, actual)

    @unittest.skip
    def testStartup(self):
        host = start_host("test_interaction", [Left, Right])
        host.put("/app/balance/right/weigh", None, 5)
        print(host.get("/app/balance/right/weight"))


if __name__ == "__main__":
    unittest.main()

