"""A hosted :class:`Library` or :class:`Service`."""

import inspect
import logging
import typing

from .chain import Chain
from .collection import Collection, Column
from .collection.table import Schema as TableSchema
from .generic import Map, Tuple
from .interface import Interface
from .json import to_json
from .reflect.functions import parse_args
from .reflect.stub import MethodStub
from .scalar import Scalar
from .scalar.number import U32
from .scalar.ref import Ref, form_of, get_ref
from .scalar.value import Nil
from .state import hash_of, Class, Instance, Object, State
from .uri import URI


class _Inspector(object):
    def __init__(self, hidden=lambda name: name.startswith('_'), parent=None, filter_classes=[], filter_instances=[]):
        assert all(inspect.isclass(cls) for cls in filter_classes)
        assert all(inspect.isclass(cls) for cls in filter_instances)

        self.filter_classes = filter_classes
        self.filter_instances = filter_instances
        self.hidden = hidden
        self.parent = parent

    def attributes(self, instance):
        parent_members = dict(inspect.getmembers(self.parent)) if self.parent else {}
        members = dict(inspect.getmembers(instance))

        for name in sorted(members.keys()):
            if self.hidden(name):
                continue

            attr = members[name]

            if name in parent_members:
                if attr is parent_members[name] or attr == parent_members[name]:
                    continue
                elif hasattr(attr, "__code__") and hasattr(parent_members[name], "__code__"):
                    if attr.__code__ is parent_members[name].__code__:
                        continue
            elif inspect.ismethod(attr) and inspect.isclass(attr.__self__):
                # it's a @classmethod
                continue
            elif inspect.isclass(attr):
                if any(issubclass(attr, cls) for cls in self.filter_classes):
                    continue
            elif any(isinstance(attr, cls) for cls in self.filter_instances):
                continue

            form = attr
            while isinstance(form, State):
                form = form_of(form)

            if any(isinstance(form, cls) for cls in self.filter_instances):
                continue

            yield name, attr


class Meta(type):
    """The metaclass of a :class:`Model` which provides support for `form_of` and `to_json`."""

    def parents(cls):
        parents = []
        for parent in cls.mro()[1:]:
            if issubclass(parent, State):
                if URI(parent) != URI(cls):
                    parents.append(parent)

        return parents

    def __form__(cls):
        parents = [c for c in cls.parents() if not issubclass(c, Model)]

        instance = cls(form=URI("self"))
        instance_header = get_ref(instance, "self")

        form = {}
        inspector = _Inspector(parent=parents[0], filter_instances=[URI]) if parents else _Inspector()
        for name, attr in inspector.attributes(instance):
            if isinstance(attr, MethodStub):
                for method_name, method in attr.expand(instance_header, name):
                    form[method_name] = to_json(method)
            else:
                form[name] = attr

        return form

    def __json__(cls):
        parents = cls.parents()

        if not parents or URI(parents[0]).startswith(URI(State)):
            return {str(URI(Class)): to_json(form_of(cls))}
        else:
            return {str(URI(parents[0])): to_json(form_of(cls))}


class Model(Object, metaclass=Meta):
    """A data model used by a :class:`Service`"""

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

        if not hasattr(self, "__uri__"):
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
            else:
                pass  # self.name is already set to attr, just leave it alone

    def __json__(self):
        if form_of(self) is self:
            raise RuntimeError(f"{self} is not JSON-encodable")

        form = form_of(self)
        form = form if form else [None]

        if isinstance(form, URI) or isinstance(form, Ref):
            return to_json(form)

        elif self.__uri__.startswith("/state"):
            raise ValueError(f"{self} has no URI defined (consider overriding the __uri__ attribute)")

        else:
            return {str(self.__uri__): to_json(form)}

    def __ns__(self, _context, _name_hint):
        logging.debug(f"will not deanonymize model {self}")

    def __ref__(self, name):
        return ModelRef(self, name)


class _Header(object):
    pass


class Dynamic(Instance):
    """A dynamic :class:`Model` whose methods can be defined according to its compile-time state"""

    def __init__(self, form=None):
        if form is not None:
            raise ValueError(f"Dynamic model {self.__class__.__name__} does not support instantiation by reference")

        if not isinstance(self, Model):
            raise TypeError(f"{self.__class__} must be a subclass of Model")

        for name, attr in _Inspector().attributes(self):
            if isinstance(attr, MethodStub):
                for method_name, method in attr.expand(self, name):
                    setattr(self, method_name, method)

    def __form__(self):
        header = ModelRef(self, "self")

        form = {}
        for name, attr in _Inspector(parent=Instance, filter_instances=[Library]).attributes(self):
            if hasattr(self.__class__, name) and isinstance(getattr(self.__class__, name), MethodStub):
                stub = getattr(self.__class__, name)
                for method_name, method in stub.expand(header, name):
                    form[method_name] = method
            elif isinstance(attr, ModelRef):
                form[name] = attr.instance
            else:
                form[name] = attr

        return form

    def __json__(self):
        form = form_of(self)
        form = form if form else [None]
        return {str(self.__uri__): to_json(form)}

    def __ns__(self, _context, _name_hint):
        logging.debug(f"will not deanonymize dynamic model {self}")

    def __repr__(self):
        return f"a Dynamic model {self.__class__.__name__}"


class ModelRef(Ref):
    """A reference to a :class:`Model`."""

    def __init__(self, instance, name):
        name = name if isinstance(name, URI) else URI(name)

        if hasattr(instance, "instance"):
            raise RuntimeError(f"the attribute name 'instance' is reserved (use a different name in {instance})")

        self.instance = instance
        self.__uri__ = name if isinstance(name, URI) else URI(name)

        for name, attr in _Inspector(lambda name: name.startswith("__"), type(self)).attributes(self.instance):
            if hasattr(attr, "__ref__"):
                setattr(self, name, get_ref(attr, self.__uri__.append(name)))
            else:
                setattr(self, name, attr)

    def __hash__(self):
        return hash_of(self.instance)

    def __json__(self):
        return to_json(self.__uri__)

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
    """A stateless set of related functions"""

    def __init__(self):
        self._methods = {}

        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue

            if isinstance(attr, MethodStub):
                self._methods[name] = attr
                for method_name, method in attr.expand(self, name):
                    setattr(self, method_name, method)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__uri__})"

    def __json__(self):
        form = {}
        for name, attr in _Inspector(filter_classes=[Library, Dynamic], filter_instances=[Library]).attributes(self):
            if _is_mutable(attr):
                raise RuntimeError(f"{self.__class__.__name__} is a Library and must not contain mutable state")
            else:
                form[name] = to_json(attr)

        return {str(self.__uri__): form}


class Service(Library):
    """A set of related methods responsible for managing associated state"""

    def __init__(self):
        Library.__init__(self)

        for name, attr in inspect.getmembers(self, _is_mutable):
            if isinstance(attr, Chain):
                attr.__uri__ = self.__uri__.append(name)
            else:
                raise RuntimeError(f"{attr} must be managed by a Chain")

    def __json__(self):
        if not hasattr(self, "_methods"):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} has not reflected over its methods--did you forget to call Service.__init__?")

        header = _Header()
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue
            elif hasattr(attr, "__ref__"):
                setattr(header, name, get_ref(attr, f"self/{name}"))
            else:
                setattr(header, name, attr)

        form = {}
        for name, attr in _Inspector(filter_classes=[Library, Dynamic], filter_instances=[Library]).attributes(self):
            if name in self._methods:
                for method_name, method in self._methods[name].expand(header, name):
                    form[method_name] = to_json(method)

            elif _is_mutable(attr):
                assert isinstance(attr, Chain)
                chain_type = type(attr)
                collection = form_of(attr)

                if isinstance(collection, Collection):
                    schema = form_of(collection)
                    form[name] = {str(URI(chain_type)): [{str(URI(type(collection))): [to_json(schema)]}]}
                elif isinstance(collection, (dict, list, tuple, Map, Tuple)):
                    form[name] = {str(URI(chain_type)): [to_json(collection)]}
                else:
                    raise TypeError(f"invalid subject for Chain: {collection}")

            else:
                form[name] = to_json(attr)

        return {str(self.__uri__): form}


def class_name(class_or_instance):
    """A snake case representation of the class name. Accepts a class or instance as an argument."""

    if isinstance(class_or_instance, type):
        name = class_or_instance.__name__
    else:
        name = class_or_instance.__class__.__name__

    return "".join(["_" + n.lower() if n.isupper() else n for n in name]).lstrip("_")


def create_schema(modelclass: typing.Type[Model]) -> TableSchema:
    """
    Create a table schema for the given model.

    A key for the table is generated using the `class_name` function, then suffixed with '_id'.
    Each attribute of the model will be interpreted as a column if it is of type :class:`Column` or :class:`Model`.
    """

    def key(cls):
        return Column(class_name(cls) + "_id", U32)

    values = []
    indices = []
    base_attributes = set()

    for b in modelclass.__bases__:
        base_attributes |= set(dir(b))

    for f in base_attributes ^ set(dir(modelclass)):
        attr = getattr(modelclass, f)
        if isinstance(attr, Column):
            values.append(attr)
        elif inspect.isclass(attr):
            assert issubclass(attr, Model)
            values.append(key(attr))
            indices.append((class_name(attr), [key(attr).name]))

    schema = TableSchema([key(modelclass)], values)
    for i in indices:
        schema.create_index(*i)

    return schema


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


# TODO: delete
def write_config(lib, config_path, overwrite=False):
    """Write the configuration of the given :class:`Library` or :class:`Service` to the given path."""

    if inspect.isclass(lib):
        raise ValueError(f"write_config expects an instance of Library, not a class: {lib}")

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
