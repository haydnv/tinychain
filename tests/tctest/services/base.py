import os
import rjwt
import tinychain as tc

from ..process import DEFAULT_PORT, start_host


# TODO: define a generic class ServiceTest(unittest.TestCase) with an install_deps(for_service):... method


class PersistenceTest(object):
    CACHE_SIZE = "5K"
    NUM_HOSTS = 3

    def service(self, chain_type):
        raise NotImplementedError

    def execute(self, hosts):
        raise NotImplementedError

    def testBlockChain(self):
        self._execute(tc.chain.Block)

    def testSyncChain(self):
        self._execute(tc.chain.Sync)

    def _execute(self, chain_type):
        service = self.service(chain_type)

        namespace = tc.URI(service).path()[1:-2]

        key = os.urandom(32)

        hosts = []
        for i in range(self.NUM_HOSTS):
            port = DEFAULT_PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.URI(service).path()
            host = start_host(
                namespace,
                host_uri=host_uri,
                symmetric_key=key,
                cache_size=self.CACHE_SIZE)

            hosts.append(host)

        hosts[0].install(service)

        self.execute(hosts)

        for host in hosts:
            host.stop()
