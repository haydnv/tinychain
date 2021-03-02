"""Base class of a hosted service."""

import inspect

from .decorators import *
from .ref import OpRef
from .reflect import gen_headers
from .state import Op, State, Tuple
from .util import form_of, ref as get_ref, uri, URI, to_json


class Meta(type):
    """The metaclass of a :class:`State`."""

    def __form__(cls):
        class Header(cls):
            pass

        header = Header(URI("self"))
        instance = cls(URI("self"))
        parent_members = dict(inspect.getmembers(Cluster(URI("self"))))

        for name, attr in inspect.getmembers(instance):
            if name.startswith('_'):
                continue

            if isinstance(attr, MethodStub):
                setattr(header, name, attr.method(instance, name))
            elif isinstance(attr, State):
                setattr(header, name, type(attr)(URI(f"self/{name}")))
            else:
                setattr(header, name, attr)

        form = {}
        for name, attr in inspect.getmembers(instance):
            if name.startswith('_'):
                continue
            elif name in parent_members:
                if hasattr(attr, "__code__") and hasattr(parent_members[name], "__code__"):
                    if attr.__code__ is parent_members[name].__code__:
                        continue
                elif attr is parent_members[name] or attr == parent_members[name]:
                    continue

            if isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(header, name))
            else:
                form[name] = attr

        return form

    def __json__(cls):
        return {str(uri(cls)): to_json(form_of(cls))}


class Cluster(object, metaclass=Meta):
    """A hosted Tinychain service."""

    @classmethod
    def __use__(cls):
        instance = cls()
        gen_headers(instance)
        return instance

    def __init__(self, form=None):
        self.__form__ = form if form else uri(self)
        self._configure()

    def _configure(self):
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

