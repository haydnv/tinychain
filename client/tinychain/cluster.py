"""Base class of a hosted service."""

import inspect
import logging

from tinychain import ref
from tinychain.decorators import *
from tinychain.op import Op
from tinychain.reflect.meta import Meta
from tinychain.state import State, Tuple
from tinychain.util import form_of, uri, URI, to_json


class Cluster(object, metaclass=Meta):
    """A hosted TinyChain service."""

    # TODO: get rid of this
    @classmethod
    def __use__(cls):
        instance = cls()
        for name, attr in inspect.getmembers(instance):
            if name.startswith('_'):
                continue

            if isinstance(attr, MethodStub):
                setattr(instance, name, attr.method(instance, name))
            elif inspect.isclass(attr) and issubclass(attr, State):
                if not uri(attr).startswith(uri(cls)):
                    logging.warning(f"cluster at {uri(cls)} serves class with different URI: {uri(attr)}")

                @get_method
                def ctr(self, form) -> attr:
                    return attr(form)

                setattr(instance, name, ctr.method(instance, name))
            elif isinstance(attr, State):
                setattr(instance, name, type(attr)(uri(cls) + f"/{name}"))

        return instance

    def __init__(self, form=None):
        if isinstance(form, URI):
            self.__uri__ = form

        self.__form__ = form if form else uri(self).path()
        self._configure()

    def _configure(self):
        """Initialize this `Cluster`'s :class:~`chain.Chain`s."""
        pass

    def _method(self, path):
        subject = uri(self) + path
        if subject.startswith("/state") and subject.path() != uri(self.__class__):
            raise ValueError(
                f"cannot call instance method {path} with an absolute path {uri(subject)}")

        return subject

    def authorize(self, scope):
        """Raise an error if the current transaction has not been granted the given scope."""

        return ref.Get(uri(self) + "/authorize", scope)

    def grant(self, scope, op: Op, context={}):
        """Execute the given `op` after granting it the given `scope`."""

        params = {
            "scope": scope,
            "op": op,
        }

        if context:
            params["context"] = context

        return ref.Post(uri(self) + "/grant", params)

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
                logging.warning(f"invalid JSON at {config_path}: {e}")
                pass

        raise RuntimeError(f"There is already an entry at {config_path}")
    else:
        import os

        if not config_path.parent.exists():
            os.makedirs(config_path.parent)

        with open(config_path, 'w') as cluster_file:
            cluster_file.write(json.dumps(config, indent=4))

