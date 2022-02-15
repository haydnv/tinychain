"""A hosted :class:`App` or :class:`Library`."""

import inspect
import logging
import typing

from collections import OrderedDict

from tinychain.decorators import MethodStub
from tinychain.reflect import header, is_ref
from tinychain.state.generic import Tuple
from tinychain.state.ref import Ref
from tinychain.state import Class, Instance, Object, Scalar, State
from tinychain.util import deanonymize, form_of, get_ref, to_json, uri, URI


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
        parent_members = dict(inspect.getmembers(parents[0](form=URI("self")))) if parents else {}

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

        if not parents or uri(parents[0]).startswith(uri(State)):
            return {str(uri(Class)): to_json(form_of(cls))}
        else:
            return {str(uri(parents[0])): to_json(form_of(cls))}


class Model(Object, metaclass=_Meta):
    def __new__(cls, *args, **kwargs):
        if "form" in kwargs:
            return Instance.__new__(cls)
        else:
            return Class.__new__(cls)

    def __init__(self, form):
        Object.__init__(self, form)  # this will generate method headers

        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue
            elif inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue

            if inspect.isclass(attr):
                if issubclass(attr, Model):
                    raise NotImplementedError("Model with Model attribute")
                elif issubclass(attr, State):
                    setattr(self, name, attr(form=URI("self").append(name)))
                else:
                    raise TypeError(f"Model does not support attributes of type {attr}")
            elif typing.get_origin(attr) is tuple:
                setattr(self, name, Tuple.expect(attr)(URI("self").append(name)))
            else:
                pass  # self.name is already set to attr, just leave it along

    def __json__(self):
        if isinstance(form_of(self), URI) or isinstance(form_of(self), Ref):
            return to_json(form_of(self))
        else:
            return {str(uri(self)): to_json(form_of(self))}


class ModelRef(object):
    def __init__(self, instance, name):
        self.instance = instance
        self.__uri__ = URI(name)

    def __getattr__(self, name):
        if not hasattr(self.instance, name):
            raise AttributeError(f"{self.instance} has no attribute {name}")

        attr = getattr(self.instance, name)
        if isinstance(attr, MethodStub):
            return attr.method(self, name)
        else:
            return get_ref(attr, uri(self).append(name))

    def __json__(self):
        return to_json(uri(self))

    def __ref__(self, name):
        return ModelRef(self.instance, name)


def model(cls):
    class _Model(cls, metaclass=_Meta):
        def __init__(self, *args, **kwargs):
            if "form" in kwargs:
                form = kwargs["form"]
                Model.__init__(self, form)
            else:
                params = _parse_init_args(inspect.signature(cls.__init__), args, kwargs)
                cls.__init__(self, *args, **kwargs)
                Instance.__init__(self, params)

    return _Model


class Library(object):
    @staticmethod
    def exports():
        return []

    def __init__(self, form=None):
        if form is not None:
            self.__uri__ = form

        for cls in self.exports():
            setattr(self, cls.__name__, model(cls))

    def __json__(self):
        form = {}

        for cls in self.exports():
            if not issubclass(cls, Model):
                raise ValueError(f"Library can only export a Model class, not {cls}")

        for name, attr in inspect.getmembers(self):
            if name.startswith('_') or name == "exports":
                continue

            _, instance_header = header(type(self))

            if _is_mutable(attr):
                raise RuntimeError(f"{self.__class__.__name__} may not contain mutable state")
            elif hasattr(attr, "hidden") and attr.hidden:
                continue
            elif isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(instance_header, name))
            else:
                form[name] = to_json(attr)

        return {str(uri(self)): form}


def _is_mutable(state):
    if not isinstance(state, State):
        return False

    if isinstance(state, Scalar):
        return False,

    return True


def write_config(lib, config_path, overwrite=False):
    """Write the configuration of the given :class:`tc.App` or :class:`Library` to the given path."""

    if inspect.isclass(lib):
        raise ValueError(f"write_app expects an instance of App, not a class: {lib}")

    import json
    import pathlib

    config = to_json(lib)
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


def _parse_init_args(sig, args, kwargs):
    params = {}
    sig = OrderedDict(sig.parameters)
    if list(sig.keys())[0] != "self":
        raise TypeError(f"__init__ signature {sig} must begin with a 'self' parameter")

    i = 0
    for name, param in list(sig.items())[1:]:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            if args:
                raise TypeError("Model does not support variable positional arguments (*args) in __init__; " +
                                f"use keyword arguments instead")

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

    return params
