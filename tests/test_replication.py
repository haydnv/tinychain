import tinychain as tc
import unittest

from testutils import PORT, PersistenceTest


class ChainTests(PersistenceTest, unittest.TestCase):
    NAME = "replication"

    def cluster(self, chain_type):
        class Rev(tc.Cluster):
            __uri__ = tc.URI(f"http://127.0.0.1:{PORT}/app/test/replication")

            def _configure(self):
                self.rev = chain_type(tc.UInt(0))

            @tc.get_method
            def version(self) -> tc.Number:
                return self.rev

            @tc.post_method
            def bump(self, txn):
                txn.rev = self.version()
                return self.rev.set(txn.rev + 1)

        return Rev

    def execute(self, hosts):
        cluster_path = "/app/test/replication"

        # check that the replica set is correctly updated across the cluster
        expected = set("http://" + tc.uri(host) + cluster_path for host in hosts)
        for host in hosts:
            actual = {}
            for link in host.get(cluster_path + "/replicas"):
                actual.update(link)
            actual = set(actual.keys())

            self.assertEqual(expected, actual)

        # test a distributed write
        hosts[-1].post(cluster_path + "/bump")
        for host in hosts:
            actual = host.get(cluster_path + "/rev")
            self.assertEqual(actual, 1)

        # test a commit with one failed host
        hosts[-1].stop()
        hosts[-2].post(cluster_path + "/bump")
        for host in hosts[:-1]:
            actual = host.get(cluster_path + "/rev")
            self.assertEqual(actual, 2)

        # test restarting the failed host
        hosts[-1].start()
        actual = hosts[-1].get(cluster_path + "/rev")
        self.assertEqual(actual, 2)

        # test a distributed write after recovering
        hosts[0].post(cluster_path + "/bump")

        for host in hosts:
            actual = host.get(cluster_path + "/rev")
            self.assertEqual(actual, 3)


if __name__ == "__main__":
    unittest.main()

