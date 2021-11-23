import os
import shutil
import time
import tinychain as tc


TC_PATH = os.getenv("TC_PATH", "host/target/debug/tinychain")
PORT = 8702


def start_host(name, clusters=[], overwrite=True, host_uri=None, cache_size="5K", wait_time=1, timeout=30):
    print(f"start_host at {host_uri}")

    if not os.path.isfile(TC_PATH):
        raise RuntimeError(f"invalid executable path: {TC_PATH}")

    port = PORT
    if host_uri is not None and host_uri.port():
        port = host_uri.port()
    elif clusters and tc.uri(clusters[0]).port():
        port = tc.uri(clusters[0]).port()

    print(f"start_host on port {port}")

    config = []
    for cluster in clusters:
        cluster_config = f"config/{name}"
        cluster_config += str(tc.uri(cluster).path())

        tc.write_cluster(cluster, cluster_config, overwrite)
        config.append(cluster_config)

    data_dir = f"/tmp/tc/tmp/{port}/{name}"
    if overwrite and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    process = tc.host.Process(
        TC_PATH,
        workspace=f"/tmp/tc/tmp/{port}/{name}",
        data_dir=data_dir,
        clusters=config,
        port=port,
        log_level="debug",
        cache_size=cache_size,
        force_create=True,
        request_ttl=timeout)

    print(f"start host on port {port}")
    process.start(wait_time)
    return tc.host.Local(process, f"http://{process.ADDRESS}:{port}")


class PersistenceTest(object):
    CACHE_SIZE = "5K"
    NUM_HOSTS = 4
    NAME = "persistence"

    def cluster(self, chain_type):
        raise NotImplementedError

    def execute(self, hosts):
        raise NotImplementedError

    def testBlockChain(self):
        self._execute(tc.chain.Block)

    def testSyncChain(self):
        self._execute(tc.chain.Sync)

    def _execute(self, chain_type):
        name = self.NAME

        cluster = self.cluster(chain_type)

        hosts = []
        for i in range(self.NUM_HOSTS):
            port = PORT + i
            host_uri = f"http://127.0.0.1:{port}" + tc.uri(cluster).path()
            host = start_host(f"test_{name}_{i}", [cluster], host_uri=tc.URI(host_uri), cache_size=self.CACHE_SIZE)
            hosts.append(host)
            printlines(5)

        time.sleep(1)

        self.execute(hosts)

        for host in hosts:
            host.stop()


def printlines(n):
    for _ in range(n):
        print()
