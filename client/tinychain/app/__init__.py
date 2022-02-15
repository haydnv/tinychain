import inspect
import logging

from tinychain.decorators import MethodStub
from tinychain.util import form_of, to_json, uri, URI


class Library(object):
    def __form__(self):
        form = {}
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue
            elif isinstance(attr, MethodStub):
                attr = attr.method(self, name)

            form[name] = attr

        return form

    def __json__(self):
        return {str(uri(self)): to_json(form_of(self))}


def write_config(app_or_library, config_path, overwrite=False):
    """Write the configuration of the given :class:`tc.App` or :class:`Library` to the given path."""

    if inspect.isclass(app_or_library):
        raise ValueError(f"write_app expects an instance of App, not a class: {app_or_library}")

    import json
    import pathlib

    config = to_json(app_or_library)
    config_path = pathlib.Path(config_path)
    if config_path.exists() and not overwrite:
        with open(config_path) as f:
            try:
                if json.load(f) == config:
                    return
            except json.decoder.JSONDecodeError as e:
                logging.warning(f"invalid JSON at {config_path}: {e}")

        raise RuntimeError(f"there is already an entry at {config_path}")
    else:
        import os

        if not config_path.parent.exists():
            os.makedirs(config_path.parent)

        with open(config_path, 'w') as config_file:
            config_file.write(json.dumps(config, indent=4))
