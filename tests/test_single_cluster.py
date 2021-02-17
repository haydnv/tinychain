import time
import tinychain as tc
import unittest


TC_PATH = "../host/target/debug/tinychain"
PORT = 8702


class ExampleCluster(tc.Cluster, metaclass=tc.Meta):
    __ref__ = tc.URI("/app/example")
    __version__ = "0.1.0"

    def configure(self):
        self.rev = tc.Chain.Sync(tc.Number.init(0))


class ClusterTests(unittest.TestCase):
    def testStartup(self):
        host = start_host("test_startup")

        expected = 0
        actual = host.get("/app/example/rev")
        self.assertEqual(expected, actual)

    def testUpdate(self):
        host = start_host("test_update")

        def expect(n):
            actual = host.get("/app/example/rev")
            self.assertEqual(n, actual)

        expect(0)

        host.put("/app/example/rev", None, 2)
        expect(2)


        host.put("/app/example/rev", None, 4)
        expect(4)


def start_host(name, port=PORT):
    cluster_config = "../config/example"
    tc.write_cluster(ExampleCluster, cluster_config)

    host = tc.host.Local(
        workspace="/tmp/tc/tmp/" + name,
        data_dir="/tmp/tc/data/" + name,
        clusters=[cluster_config],
        force_create=True)

    host.start(TC_PATH, PORT, log_level="debug")
    return host


if __name__ == "__main__":
    unittest.main()

