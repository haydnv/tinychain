"""A hosted :class:`App` or :class:`Library`."""

import inspect
import logging
import typing

from .collection import Collection
from .reflect import parse_args
from .reflect.meta import Meta, MethodStub
from .chain import Chain
from .generic import Map, Tuple
from .interface import Interface
from .scalar.ref import depends_on, form_of, get_ref, independent, Ref
from .scalar.value import Nil
from .scalar import Scalar
from .state import Class, Instance, Object, State
from .uri import uri, URI
from .context import to_json


class Model(Object, metaclass=Meta):
    def __new__(cls, *args, **kwargs):
        if issubclass(cls, Dynamic):
            return Instance.__new__(cls)
        elif "form" in kwargs:
            return Instance.__new__(_model(cls))
        else:
            return Class.__new__(_model(cls))

    def __init__(self, form):
        if form is None:
            raise ValueError(f"form of {self} cannot be None; consider Nil instead")

        Object.__init__(self, form)  # this will generate method headers

        if not uri(self):
            raise ValueError(f"{self} has no URI defined (consider setting the __uri__ attribute)")

        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue
            elif inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue

            if inspect.isclass(attr):
                if issubclass(attr, State):
                    setattr(self, name, attr(form=URI("self").append(name)))
                elif issubclass(attr, Interface):
                    cls = type(f"{attr.__name__}State", (State, attr), {})
                    setattr(self, name, cls(form=URI("self").append(name)))
                else:
                    raise TypeError(f"Model does not support attributes of type {attr}")
            elif typing.get_origin(attr) is tuple:
                setattr(self, name, Tuple.expect(attr)(URI("self").append(name)))
            else:
                pass  # self.name is already set to attr, just leave it alone

    def __json__(self):
        if form_of(self) is self:
            raise RuntimeError(f"{self} is not JSON-encodable")

        form = form_of(self)
        form = form if form else [None]

        if isinstance(form, URI) or isinstance(form, Ref):
            return to_json(form)

        elif uri(self).startswith("/state"):
            raise ValueError(f"{self} has no URI defined (consider overriding the __uri__ attribute)")

        else:
            return {str(uri(self)): to_json(form)}

    def __ns__(self, _context, _name_hint):
        logging.debug(f"will not deanonymize model {self}")

    def __ref__(self, name):
        return ModelRef(self, name)


class _Header(object):
    pass


class Dynamic(Instance):
    def __init__(self, form=None):
        if form is not None:
            raise ValueError(f"Dynamic model {self.__class__.__name__} does not support instantiation by reference")

        if not isinstance(self, Model):
            raise TypeError(f"{self.__class__} must be a subclass of Model")

        # TODO: deduplicate with Meta.__form__
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue
            elif hasattr(attr, "hidden") and attr.hidden:
                continue
            elif inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue

            if isinstance(attr, MethodStub):
                setattr(self, name, attr.method(self, name))
            elif isinstance(attr, State):
                if not independent(attr):
                    classname = self.__class__.__name__
                    raise ValueError(f"{attr} in {classname} depends on anonymous state {depends_on(attr)}")

    # TODO: deduplicate with Meta.__form__
    def __form__(self):
        parent_members = dict(inspect.getmembers(Instance))

        header = ModelRef(self, "self")

        form = {}
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue
            elif hasattr(attr, "hidden") and attr.hidden:
                continue
            elif isinstance(attr, Library):
                # it's an external dependency
                continue
            elif inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue
            elif name in parent_members:
                if attr is parent_members[name] or attr == parent_members[name]:
                    continue
                elif hasattr(attr, "__code__") and hasattr(parent_members[name], "__code__"):
                    if attr.__code__ is parent_members[name].__code__:
                        logging.debug(f"{attr} is identical to its parent, won't be defined explicitly in {self}")
                        continue

            if hasattr(self.__class__, name) and isinstance(getattr(self.__class__, name), MethodStub):
                stub = getattr(self.__class__, name)
                form[name] = stub.method(header, name)
            elif isinstance(attr, ModelRef):
                form[name] = attr.instance
            else:
                form[name] = attr

        return form

    def __json__(self):
        form = form_of(self)
        form = form if form else [None]
        return {str(uri(self)): to_json(form)}

    def __ns__(self, _context, _name_hint):
        logging.debug(f"will not deanonymize dynamic model {self}")

    def __repr__(self):
        return f"a Dynamic model {self.__class__.__name__}"


class ModelRef(Ref):
    def __init__(self, instance, name):
        assert name

        if hasattr(instance, "instance"):
            raise RuntimeError(f"the attribute name 'instance' is reserved (use a different name in {instance})")

        self.instance = instance
        self.__uri__ = URI(name)

        # TODO: deduplicate with Meta.__form__
        for name, attr in inspect.getmembers(self.instance):
            if name.startswith('__'):
                continue
            elif inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue

            if name.startswith('_'):
                if isinstance(attr, State):
                    logging.warning(f"referencing {instance} without referencing hidden State {name}")

                setattr(self, name, attr)
            elif hasattr(instance.__class__, name) and isinstance(getattr(instance.__class__, name), MethodStub):
                stub = getattr(instance.__class__, name)
                setattr(self, name, stub.method(self, name))
            else:
                setattr(self, name, get_ref(attr, uri(self).append(name)))

    def __json__(self):
        return to_json(uri(self))

    def __ref__(self, name):
        return ModelRef(self.instance, name)


def _model(cls):
    if not issubclass(cls, Model):
        raise TypeError(f"expected a subclass of Model but found {cls}")

    class _Model(cls):
        def __init__(self, *args, **kwargs):
            if "form" in kwargs:
                form = kwargs["form"]

                if isinstance(cls, Dynamic):
                    raise RuntimeError(f"Dynamic model cannot be instantiated by reference (got {form})")

                Model.__init__(self, form)
            elif cls.__init__ is Model.__init__:
                cls.__init__(self, form=Nil())
            else:
                sig = list(inspect.signature(cls.__init__).parameters.items())
                if sig[0][0] != "self":
                    raise TypeError(f"__init__ signature {sig} must begin with a 'self' parameter")

                params = parse_args(sig[1:], *args, **kwargs)
                Instance.__init__(self, params)

        def __json__(self):
            return Model.__json__(self)

    _Model.__name__ = cls.__name__
    return _Model


class Library(object):
    def __init__(self):
        self._methods = {}

        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue

            if isinstance(attr, MethodStub):
                self._methods[name] = attr
                setattr(self, name, attr.method(self, name))

    def __repr__(self):
        return f"{self.__class__.__name__}({uri(self)})"

    # TODO: deduplicate with Meta.__json__
    def __json__(self):
        form = {}
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue

            if inspect.isclass(attr):
                if issubclass(attr, Library) or issubclass(attr, Dynamic):
                    continue
            elif isinstance(attr, Library):
                continue

            if hasattr(attr, "hidden") and attr.hidden:
                continue
            elif inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue
            elif _is_mutable(attr):
                raise RuntimeError(f"{self.__class__.__name__} may not contain mutable state")
            else:
                form[name] = to_json(attr)

        return {str(uri(self)): form}


class App(Library):
    def __init__(self):
        Library.__init__(self)

        for name, attr in inspect.getmembers(self, _is_mutable):
            if isinstance(attr, Chain):
                attr.__uri__ = uri(self).append(name)
            else:
                raise RuntimeError(f"{attr} must be managed by a Chain")

    def __json__(self):
        if not hasattr(self, "_methods"):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} has not reflected over its methods--did you forget to call App.__init__?")

        header = _Header()
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue
            elif isinstance(attr, MethodStub):
                setattr(header, name, attr.method(self, name))
            else:
                setattr(header, name, get_ref(attr, f"self/{name}"))

        # TODO: deduplicate with Library.__json__ and Meta.__json__
        form = {}
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue

            if inspect.isclass(attr):
                if issubclass(attr, Library) or issubclass(attr, Dynamic):
                    continue
            elif isinstance(attr, Library):
                continue

            if hasattr(attr, "hidden") and attr.hidden:
                continue
            elif inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue

            elif name in self._methods:
                form[name] = to_json(self._methods[name].method(header, name))

            elif _is_mutable(attr):
                assert isinstance(attr, Chain)
                chain_type = type(attr)
                collection = form_of(attr)

                if isinstance(collection, Collection):
                    schema = form_of(collection)
                    form[name] = {str(uri(chain_type)): [{str(uri(type(collection))): [to_json(schema)]}]}
                elif isinstance(collection, (dict, list, tuple, Map, Tuple)):
                    form[name] = {str(uri(chain_type)): [to_json(collection)]}
                else:
                    raise TypeError(f"invalid subject for Chain: {collection}")

            else:
                form[name] = to_json(attr)

        return {str(uri(self)): form}


def dependencies(lib_or_model):
    deps = []
    for name, attr in inspect.getmembers(lib_or_model):
        if name.startswith("_"):
            continue

        if isinstance(attr, Library):
            deps.extend(dependencies(attr))
        elif inspect.isclass(attr) and issubclass(attr, Model):
            deps.extend(dependencies(attr))

    if isinstance(lib_or_model, Library):
        deps.append(lib_or_model)

    return deps


def write_config(lib, config_path, overwrite=False):
    """Write the configuration of the given :class:`tc.App` or :class:`Library` to the given path."""

    if inspect.isclass(lib):
        raise ValueError(f"write_app expects an instance of Library, not a class: {lib}")

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
            assert config_path.parent.exists()

        with open(config_path, 'w') as config_file:
            config_file.write(json.dumps(config, indent=4))


def _is_mutable(state):
    if not isinstance(state, State):
        return False

    if isinstance(state, Scalar):
        return False

    return True
