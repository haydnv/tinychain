import time
import tinychain as tc

from ..process import DEFAULT_PORT, start_host


class PersistenceTest(object):
    CACHE_SIZE = "5K"
    NUM_HOSTS = 4

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

        hosts = []
        for i in range(self.NUM_HOSTS):
            port = DEFAULT_PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.URI(service).path()
            host = start_host(f"test_{service.NAME}_{i}", [], host_uri=host_uri, cache_size=self.CACHE_SIZE)
            hosts.append(host)

        hosts[0].put(tc.URI(tc.app.Service), str(service.NS)[1:], tc.URI(service)[:-2])
        hosts[0].put(tc.URI(service).path()[:-2], tc.URI(service)[-2], service)

        time.sleep(1)

        self.execute(hosts)

        for host in hosts:
            host.stop()
