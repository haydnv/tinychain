import json
import unittest
import tinychain as tc


CONSERVED = tc.Number(20)


class Balance(tc.Cluster):
    __uri__ = tc.URI("/app/balance")

    def _configure(self):
        self.weight = tc.Chain.Sync(tc.Number(0))


class Left(Balance):
    __uri__ = tc.uri(Balance) + "/left"

    @tc.put_method
    def weigh(self, txn, key: tc.Nil, weight: tc.Number):
        right = tc.use(Right)
        txn.total = CONSERVED
        txn.update = tc.After(right.weigh(None, txn.total - weight), self.weight.set(weight))
        return tc.If(self.weight == weight, None, txn.update)


class Right(Balance):
    __uri__ = tc.uri(Balance) + "/right"

    @tc.put_method
    def weigh(self, txn, key: tc.Nil, weight: tc.Number):
        left = tc.use(Left)
        txn.total = CONSERVED
        txn.update = tc.After(left.weigh(None, txn.total - weight), self.weight.set(weight))
        return tc.If(self.weight == weight, None, txn.update)


class ClusterTests(unittest.TestCase):
    def testToJson(self):
        self.maxDiff = None
        expected = {
            'weigh': {'/state/scalar/op/put': ['key', 'weight', [
                ['total', 20],
                ['update', {'/state/scalar/ref/after': [
                    {'/app/balance/right/weigh': [None, {'$total/sub': [{'$weight': []}]}]}, 
                    {'$self/weight': [None, {'$weight': []}]}
                ]}],
                ['_return', {'/state/scalar/ref/if': [
                    {'$weight/eq': [{'$self/weight': [None]}]},
                    None,
                    {'$update': []}
                ]}]
            ]]},
            'weight': {'/state/chain/sync': [0]}
        }

        actual = tc.to_json(tc.form_of(Left))
        print(actual)
        return
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

