import tinychain as tc
import unittest


class ExampleCluster(tc.Cluster, metaclass=tc.Meta):
    __ref__ = tc.URI("/app/example")

    def configure(self):
        self.rev = tc.Chain.Sync(tc.Number.init(0))

    @tc.get_method
    def current(self) -> tc.Number:
        return self.rev.subject()

    @tc.get_method
    def history(self) -> tc.Chain:
        return tc.OpRef.Get(self.rev)

    @tc.post_method
    def bump(self, cxt, version: tc.Number):
        return tc.If(
            version > self.current(),
            tc.OpRef.Put(self.rev, None, version),
            tc.error.BadRequest("Version too old"))


class DownstreamCluster(tc.Cluster, metaclass=tc.Meta):
    __ref__ = tc.URI("/app/downstream")

    @tc.get_method
    def example(self):
        example = tc.use(ExampleCluster)
        return example.current()


class ClusterTests(unittest.TestCase):
    def testToJson(self):
        expected = {
            '/app/example': {
                'bump': {'/state/scalar/op/post': [
                    ['_return', {'/state/scalar/ref/if': [
                        {'$version/gt': [{'$self/current': [None]}]},
                        {'$self/rev': [None, {"$version": []}]},
                        {"/error/bad_request": "Version too old"}
                    ]}]
                ]},
                'current': {
                    '/state/scalar/op/get': ['key', [
                        ['_return', {'$self/rev/subject': [None]}]
                    ]]
                },
                'history': {
                    '/state/scalar/op/get': ['key', [
                        ['_return', {'$self/rev': [None]}]
                    ]]
                },
                'rev': {'/state/chain/sync': [{'/state/scalar/value/number': [0]}]}
            }
        }

        actual = tc.to_json(ExampleCluster)
        self.assertEqual(expected, actual)

    def testDownstream(self):
        expected = {
            '/app/downstream': {
                'example': {
                    '/state/scalar/op/get': ['key', [
                        ['_return', {'/app/example/current': [None]}]
                    ]]
                }
            }
        }

        actual = tc.to_json(DownstreamCluster)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()

