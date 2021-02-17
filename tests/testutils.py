import tinychain as tc


TC_PATH = "host/target/debug/tinychain"
PORT = 8702


def start_host(name, clusters, port=PORT):
    config = []
    for cluster in clusters:
        cluster_config = "config" + str(tc.uri(cluster))
        tc.write_cluster(cluster, cluster_config)
        config.append(cluster_config)

    host = tc.host.Local(
        workspace="/tmp/tc/tmp/" + name,
        data_dir="/tmp/tc/data/" + name,
        clusters=config,
        force_create=True)

    host.start(TC_PATH, PORT, log_level="debug")
    return host

