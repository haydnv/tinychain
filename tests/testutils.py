import os
import shutil
import time
import tinychain as tc


TC_PATH = "host/target/debug/tinychain"
PORT = 8702


def start_host(name, clusters=[], overwrite=True, host_uri=None):
    port = PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()
    elif clusters and tc.uri(clusters[0]).port():
        port = tc.uri(clusters[0]).port()

    config = []
    for cluster in clusters:
        cluster_config = f"config/{name}"
        cluster_config += str(tc.uri(cluster).path())

        tc.write_cluster(cluster, cluster_config, overwrite)
        config.append(cluster_config)

    data_dir = f"/tmp/tc/tmp/{port}/{name}"
    if overwrite and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    host = tc.host.Local(
        TC_PATH,
        workspace=f"/tmp/tc/tmp/{port}/{name}",
        data_dir=data_dir,
        clusters=config,
        port=port,
        log_level="debug",
        cache_size="5K",
        force_create=True)

    print(f"start host on port {port}")
    host.start()
    return host


class PersistenceTest(object):
    NUM_HOSTS = 4
    NAME = "persistence"

    def cluster(self, chain_type):
        raise NotImplementedError

    def execute(self, hosts):
        raise NotImplementedError

    def testBlockChain(self):
        self._execute(tc.Chain.Block)

    def testSyncChain(self):
        self._execute(tc.Chain.Sync)

    def _execute(self, chain_type):
        name = self.NAME

        cluster = self.cluster(chain_type)
        
        hosts = []
        for i in range(self.NUM_HOSTS):
            port = PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.uri(cluster).path()
            host = start_host(f"test_{name}_{i}", [cluster], host_uri=tc.URI(host_uri))
            hosts.append(host)
            printlines(5)

        time.sleep(1)

        self.execute(hosts)

        for host in hosts:
            host.stop()


def printlines(n):
    for _ in range(n):
        print()

