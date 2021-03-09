import os
import shutil
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

    data_dir = "/tmp/tc/tmp/" + name
    if overwrite and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    print(f"start host on port {port}")
    return tc.host.Local(
        TC_PATH,
        workspace="/tmp/tc/tmp/" + name,
        data_dir=data_dir,
        clusters=config,
        port=port,
        log_level="debug",
        force_create=True)

