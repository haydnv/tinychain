from .annotations import *
from .reflect import gen_headers
from .state import OpRef, State
from .util import form_of, ref as get_ref, uri, URI, to_json


class Cluster(object):
    @classmethod
    def __use__(cls):
        instance = cls()
        gen_headers(instance)
        return instance

    def __init__(self, form=None):
        self.__form__ = form if form else uri(self)
        self.configure()

    def configure(self):
        pass

    def authorize(self, scope):
        return OpRef.Get(uri(self) + "/authorize", scope)

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

    config = {str(uri(cluster)): to_json(form_of(cluster))}
    config_path = pathlib.Path(config_path)
    if config_path.exists() and not overwrite:
        with open(config_path) as f:
            try:
                if json.load(f) == config:
                    return
            except json.decoder.JSONDecodeError as e:
                print(f"warning: invalid JSON at {config_path}: {e}")
                pass

        raise RuntimeError(f"There is already an entry at {config_path}")
    else:
        import os

        if not config_path.parent.exists():
            os.makedirs(config_path.parent)

        with open(config_path, 'w') as cluster_file:
            cluster_file.write(json.dumps(config, indent=4))

