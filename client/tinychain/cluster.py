from .annotations import *
from .reflect import gen_headers
from .state import OpRef, State
from .util import ref as get_ref, uri, URI, to_json


class Cluster(object):
    @classmethod
    def __use__(cls):
        instance = cls()
        gen_headers(instance)
        return instance

    def __init__(self, ref=None):
        if ref is None:
            ref = get_ref(self.__class__)

        self.__ref__ = ref

        self.configure()

    def configure(self):
        pass

    @get_method
    def authorize(self, txn, scope):
        pass

    def grant(self, scope, op, context={}):
        params = {
            "scope": scope,
            "op": op,
        }

        if context:
            params["context"] = context

        return OpRef.Post(uri(self) + "/grant", **params)

    @put_method
    def install(self):
        pass


def write_cluster(cluster, config_path, overwrite=False):
    import json
    import pathlib

    config_path = pathlib.Path(config_path)
    if config_path.exists() and not overwrite:
        with open(config_path) as f:
            try:
                if json.load(f) == to_json(cluster):
                    return
            except json.decoder.JSONDecodeError:
                pass

        raise RuntimeError(f"There is already an entry at {config_path}")
    else:
        import os

        if not config_path.parent.exists():
            os.makedirs(config_path.parent)

        with open(config_path, 'w') as cluster_file:
            cluster_file.write(json.dumps(to_json(cluster), indent=4))

