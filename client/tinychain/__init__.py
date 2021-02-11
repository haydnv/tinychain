import tinychain.error
import tinychain.host

from tinychain.annotations import *
from tinychain.chain import Chain, sync_chain
from tinychain.cluster import Cluster
from tinychain.reflect import Meta
from tinychain.state import Class, If, State, Scalar, OpRef
from tinychain.util import *
from tinychain.value import *


def write_cluster(cluster, config_path):
    import json
    import pathlib
    config_path = pathlib.Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            try:
                if json.load(f) == to_json(cluster):
                    return
            except json.decoder.JSONDecodeError:
                pass

        raise RuntimeError(f"There is already an entry at {cluster_path}")
    else:
        with open(config_path, 'w') as cluster_file:
            cluster_file.write(json.dumps(to_json(cluster), indent=4))

