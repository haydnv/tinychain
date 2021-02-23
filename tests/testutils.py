import os
import shutil
import tinychain as tc


TC_PATH = "host/target/debug/tinychain"
PORT = 8702


def start_host(name, clusters):
    port = PORT
    if clusters:
        port = tc.uri(clusters[0]).port()
        port = port if port else PORT

    config = []
    for cluster in clusters:
        cluster_config = f"config/{name}"

        cluster_uri = tc.uri(cluster)
        if cluster_uri.port() is not None and cluster_uri.port() != port:
            raise ValueError(f"invalid port {cluster_uri.port()}, expected {port}")

        if cluster_uri.host():
            cluster_config += f"/{cluster_uri.host()}"

        if cluster_uri.port():
            cluster_config += f"/{cluster_uri.port()}"

        cluster_config += str(cluster_uri.path())

        tc.write_cluster(cluster, cluster_config, True)
        config.append(cluster_config)

    data_dir = "/tmp/tc/tmp/" + name
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    host = tc.host.Local(
        workspace="/tmp/tc/tmp/" + name,
        data_dir=data_dir,
        clusters=config,
        force_create=True)

    print(f"start host on port {port}")
    host.start(TC_PATH, port, log_level="debug")
    return host

