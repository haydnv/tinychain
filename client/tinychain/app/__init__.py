import inspect
import logging

from collections import OrderedDict

from tinychain.chain import Chain
from tinychain.reflect import header, MethodStub
from tinychain.state.ref import Ref
from tinychain.state import Class, Instance, Object, Scalar, State
from tinychain.util import deanonymize, form_of, to_json, uri, URI


# TODO: deduplicate with reflect.Meta
class _Meta(type):
    """The metaclass of a :class:`Model` which provides support for `form_of` and `to_json`."""

    def parents(cls):
        parents = []
        for parent in cls.mro()[1:]:
            if issubclass(parent, State):
                if uri(parent) != uri(cls):
                    parents.append(parent)

        return parents

    def __form__(cls):
        parents = [c for c in cls.parents() if not issubclass(c, Model)]
        if not parents:
            raise ValueError("TinyChain class must extend a subclass of State")

        parent_members = dict(inspect.getmembers(parents[0](form=URI("self"))))

        instance, instance_header = header(cls)

        form = {}
        for name, attr in inspect.getmembers(instance):
            if name.startswith('_') or isinstance(attr, URI):
                continue
            elif name in parent_members:
                if attr is parent_members[name] or attr == parent_members[name]:
                    logging.debug(f"{attr} is identical to its parent, won't be defined explicitly in {cls}")
                    continue
                elif hasattr(attr, "__code__") and hasattr(parent_members[name], "__code__"):
                    if attr.__code__ is parent_members[name].__code__:
                        logging.debug(f"{attr} is identical to its parent, won't be defined explicitly in {cls}")
                        continue

                logging.info(f"{attr} ({name}) overrides a parent method and will be explicitly defined in {cls}")
            elif inspect.ismethod(attr) and attr.__self__ is cls:
                # it's a @classmethod
                continue

            elif isinstance(attr, State):
                while isinstance(attr, State):
                    attr = form_of(attr)

                if isinstance(attr, URI):
                    continue

                form[name] = attr
            if isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(instance_header, name))
            else:
                form[name] = attr

        return form

    def __json__(cls):
        parents = cls.parents()

        if uri(parents[0]).startswith(uri(State)):
            return {str(uri(Class)): to_json(form_of(cls))}
        else:
            return {str(uri(parents[0])): to_json(form_of(cls))}


class Model(Object, metaclass=_Meta):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("cannot initialize a Model itself")

    def __json__(self):
        if isinstance(form_of(self), URI) or isinstance(form_of(self), Ref):
            return to_json(form_of(self))
        else:
            return {str(uri(self)): to_json(form_of(self))}


class _Model(Model):
    pass


def model(cls):
    if not issubclass(cls, Model):
        raise ValueError("not a Model: {cls}")

    attrs = {}
    for name, attr in inspect.getmembers(cls):
        if name.startswith('_'):
            continue
        elif hasattr(attr, "hidden") and attr.hidden:
            continue
        elif isinstance(attr, MethodStub):
            continue
        elif hasattr(Model, name):
            continue
        elif inspect.ismethod(attr) and attr.__self__ is cls:
            # it's a @classmethod
            continue

        if not inspect.isclass(attr):
            raise TypeError(f"model attribute must be a type of State, not {attr}")

        if issubclass(attr, Model):
            attr = model(attr)
        elif not issubclass(attr, State):
            raise TypeError(f"unknown model attribute type: {attr}")

        attrs[name] = attr(form=URI("self").append(name))

    class __Model(cls):
        def __new__(*args, **kwargs):
            cls = args[0]

            if "form" in kwargs:
                return Instance.__new__(cls)
            else:
                return _Model.__new__(cls)

        def __init__(self, *args, **kwargs):
            for name in attrs:
                setattr(self, name, attrs[name])

            if "form" in kwargs:
                return Instance.__init__(self, form=kwargs["form"])

            params = {}
            sig = OrderedDict(inspect.signature(cls.__init__).parameters)
            if list(sig.keys())[0] != "self":
                raise TypeError(f"{cls}.__init__ signature must begin with a 'self' parameter")

            i = 0
            for name, param in list(sig.items())[1:]:
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    if args:
                        raise TypeError("Model does not support positional arguments in __init__; " +
                                        f"use keyword arguments in {cls} instead")
                elif param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    params[name] = args[i]
                    i += 1
                elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    if name in kwargs:
                        params[name] = kwargs[name]
                    elif i < len(args):
                        params[name] = args[i]
                        i += 1
                    elif param.default != inspect.Parameter.empty:
                        params[name] = param.default
                    else:
                        raise ValueError(
                            f"no value specified for parameter {name} in {self.__class__.__name__}.__init__")
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    params.update(kwargs)

            params = params if params else None
            Instance.__init__(self, form=params)

    return __Model


class Library(object):
    def __init__(self, form=None):
        if form is not None:
            self.__uri__ = form

        for cls in self.exports():
            if not uri(cls).startswith(uri(self.__class__)):
                raise ValueError(f"the library at {uri(self)} cannot export a class at {uri(cls)}")

            expected_uri = uri(self.__class__).append(cls.__name__)
            if uri(cls) != expected_uri:
                raise ValueError(f"class {cls.__name__} at {uri(cls)} does not match the expected URI {expected_uri}")

            setattr(self, cls.__name__, model(cls))

        self._allow_mutable = False

    def __form__(self):
        form = {}

        for cls in self.exports():
            if not issubclass(cls, Model):
                raise ValueError(f"Library can only export a Model class, not {cls}")

        for name, attr in inspect.getmembers(self):
            if name.startswith('_') or name == "exports":
                continue

            _, instance_header = header(type(self))

            if not self._allow_mutable and _is_mutable(attr):
                raise RuntimeError(f"{self.__class__.__name__} may not contain mutable state")
            if self._allow_mutable and _is_mutable(attr) and not isinstance(attr, Chain):
                raise RuntimeError("mutable state must be in a Chain")
            elif hasattr(attr, "hidden") and attr.hidden:
                continue
            elif isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(instance_header, name))
            else:
                form[name] = to_json(attr)

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
            logging.info(f"write config for {app_or_library} tp {config_path}")
            config_file.write(json.dumps(config, indent=4))


def _is_mutable(state):
    if not isinstance(state, State):
        return False

    if isinstance(state, Scalar):
        return False

    return True
