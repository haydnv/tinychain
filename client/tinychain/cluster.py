"""Base class of a hosted service."""

from .decorators import *
from .ref import OpRef
from .reflect import gen_headers
from .state import Op, State, Tuple
from .util import form_of, ref as get_ref, uri, URI, to_json


class Cluster(object):
    """A hosted Tinychain service."""

    @classmethod
    def __use__(cls):
        """Return an instance of this `Cluster` with callable methods."""

        instance = cls()
        gen_headers(instance)
        return instance

    def __init__(self, form=None):
        self.__form__ = form if form else uri(self)
        self.configure()

    def configure(self):
        """Initialize this `Cluster`'s :class:~`chain.Chain`s."""
        pass

    def authorize(self, scope):
        """Raise an error if the current transaction has not been granted the given scope."""

        return OpRef.Get(uri(self) + "/authorize", scope)

    def grant(self, scope, op: Op, context={}):
        """Execute the given `op` after granting it the given `scope`."""

        params = {
            "scope": scope,
            "op": op,
        }

        if context:
            params["context"] = context

        return OpRef.Post(uri(self) + "/grant", **params)

    @put_method
    def install(self, txn, cluster_link: URI, scopes: Tuple):
        """Trust the cluster at the given link to grant the given scopes."""
        pass


def write_cluster(cluster, config_path, overwrite=False):
    """Write the configuration of the given :class:`Cluster` to the given path."""

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

