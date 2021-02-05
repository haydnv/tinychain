import tinychain as tc
import unittest


TC_PATH = "../host/target/debug/tinychain"
PORT = 8702


class ExampleCluster(tc.Cluster, metaclass=tc.Meta):
    __ref__ = tc.URI("/app/example")
    __version__ = "0.1.0"

    def configure(self):
        self.rev = tc.sync_chain(tc.Number.init(0))


class ClusterTests(unittest.TestCase):
    def testStartup(self):

        host = tc.host.Local(
            workspace="/tmp/tc/tmp",
            data_dir="/tmp/tc/data",
            clusters=[ExampleCluster])

        host.start(TC_PATH, PORT)
        print(host.get("/app/example/rev"))


if __name__ == "__main__":
    unittest.main()

