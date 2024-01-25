"""A hosted :class:`Service` or :class:`Library`."""

import inspect
import logging

from .chain import Chain
from .collection import Collection, Column
from .interface import Interface
from .json import to_json
from .reflect.functions import parse_args
from .reflect.meta import Meta
from .reflect.stub import MethodStub
from .scalar import Scalar
from .scalar.number import U32
from .scalar.ref import Ref, form_of, get_ref, is_literal
from .scalar.value import Nil
from .state import hash_of, Class, Instance, Object, State
from .uri import URI


def class_name(class_or_instance):
    """
    Returns a snake-case representation of the class name.

    Example: `assert class_name(LargeInt) == "large_int") and class_name(LargeInt(123) == "large_int")`
    """

    cls = class_or_instance if isinstance(class_or_instance, type) else class_or_instance.__class__
    return "".join(["_" + n.lower() if n.isupper() else n for n in cls.__name__]).lstrip("_")


class Model(Object, metaclass=Meta):
    __uri__ = URI("/class")

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

    @classmethod
    def key(cls):
        """A Column object which will be used as the key for a given model."""
        return [Column(class_name(cls) + "_id", U32)]


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

            if inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue

            if isinstance(attr, MethodStub):
                for method_name, method in attr.expand(self, name):
                    setattr(self, method_name, method)

    # TODO: deduplicate with Meta.__form__
    def __form__(self):
        parent_members = dict(inspect.getmembers(Instance))

        header = ModelRef(self, "self")

        form = {}
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
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

            # TODO: resolve these in alphabetical order
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
    def __init__(self, instance, name):
        name = name if isinstance(name, URI) else URI(name)

        if hasattr(instance, "instance"):
            raise RuntimeError(f"the attribute name 'instance' is reserved (use a different name in {instance})")

        self.instance = instance
        self.__uri__ = name if isinstance(name, URI) else URI(name)

        # TODO: deduplicate with Meta.__form__
        for name, attr in inspect.getmembers(self.instance):
            if name.startswith('__'):
                continue
            elif inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue
            elif hasattr(self.__class__, name) and attr is getattr(self.__class__, name):
                # it's a class attribute
                pass
            elif hasattr(attr, "__ref__"):
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
    __uri__ = URI("/lib")

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

            if inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue
            elif _is_mutable(attr):
                raise RuntimeError(f"{self} is a Library and must not contain mutable state")
            else:
                form[name] = to_json(attr)

        return {str(URI(self)[:-1]): form}


class Service(Library):
    __uri__ = URI("/service")

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
            raise RuntimeError(f"{name} has not reflected over its methods--did you forget to call App.__init__?")

        header = _Header()
        for name, attr in inspect.getmembers(self):
            if name.startswith('_'):
                continue
            elif hasattr(attr, "__ref__"):
                setattr(header, name, get_ref(attr, f"self/{name}"))
            else:
                setattr(header, name, attr)

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

            if inspect.ismethod(attr) and attr.__self__ is self.__class__:
                # it's a @classmethod
                continue

            elif name in self._methods:
                for method_name, method in self._methods[name].expand(header, name):
                    form[method_name] = to_json(method)

            elif _is_mutable(attr):
                assert isinstance(attr, Chain)
                chain_type = type(attr)
                collection = form_of(attr)

                if isinstance(collection, Collection):
                    schema = form_of(collection)
                    form[name] = {str(URI(chain_type)): [{str(URI(type(collection))): [to_json(schema)]}]}
                else:
                    raise TypeError(f"invalid subject for Chain: {collection}")

            else:
                form[name] = to_json(attr)

        return {str(URI(self)[:-1]): form}


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


def model_uri(namespace, lib_name, version, model_name):
    assert namespace.startswith('/'), f"a namespace must be a URI path (e.g. '/name'), not {namespace}"
    assert '/' not in lib_name, f"library or service name {lib_name} must not contain a '/'"
    assert '/' not in model_name, f"model name {lib_name} must not contain a '/'"

    return (URI(Model) + namespace).extend(lib_name, version, model_name)


def library_uri(lead, namespace, name, version):
    return _make_uri(Library, lead, namespace, name, version)


def service_uri(lead, namespace, name, version):
    return _make_uri(Service, lead, namespace, name, version)


def _is_mutable(state):
    if not isinstance(state, State):
        return False

    if isinstance(state, Scalar):
        return False

    return True


def _make_uri(parent, lead, namespace, name, version):
    if lead is not None:
        assert "://" in lead, f"lead replica {lead} must specify a protocol (e.g. 'http://...')"

    assert namespace.startswith('/'), f"a namespace must be a URI path (e.g. '/name'), not {namespace}"
    assert '/' not in name, f"{parent.__name__} name {name} must not contain a '/'"

    namespace = URI(namespace)
    assert namespace.host() is None, f"namespace {namespace} must not specify a host"
    assert is_literal(version), f"version number {version} must be known at compile-time"

    uri = lead if lead else URI('/')
    uri += URI(parent)
    uri += namespace
    uri += name
    uri += version

    return uri
