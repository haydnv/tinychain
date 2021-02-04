import tinychain as tc
import unittest


@tc.class_def
class ExampleCluster(tc.Cluster):
    __ref__ = tc.URI("/app/example")

    def configure(self):
        self.rev = tc.sync_chain(tc.Number.init(0))

    @tc.get_method
    def current(self) -> tc.Number:
        return self.rev.subject()

    @tc.get_method
    def history(self) -> tc.Chain:
        return tc.OpRef.Get(self.rev)


class ClusterTests(unittest.TestCase):
    def testToJson(self):
        expected = {
            '/app/example': {
                'current': {'/state/scalar/op/get': ['key', [
                    ['_return', {'$self/rev/subject': [None]}]
                ]]},
                'history': {'/state/scalar/op/get': ['key', [
                    ['_return', {'$self/rev': [None]}]]
                ]},
                'rev': {'/state/chain/sync': [{'/state/scalar/value/number': [0]}]}
            }
        }

        actual = tc.to_json(ExampleCluster)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

