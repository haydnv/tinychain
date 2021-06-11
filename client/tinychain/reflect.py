import inspect

from pydoc import locate

from . import op, ref
from .state import State
from .util import form_of, to_json, uri, Context, URI


def gen_headers(instance):
    for name, attr in inspect.getmembers(instance):
        if name.startswith('_'):
            continue

        if isinstance(attr, MethodStub):
            setattr(instance, name, attr.method(instance, name))


class Meta(type):
    """The metaclass of a :class:`State`."""

    def __form__(cls):
        mro = cls.mro()
        if len(mro) < 2:
            raise ValueError("Tinychain class must extend a subclass of State")

        parent_members = dict(inspect.getmembers(mro[1](URI("self"))))

        class Header(cls):
            pass

        instance_uri = URI("self")
        header = Header(instance_uri)
        instance = cls(instance_uri)

        for name, attr in inspect.getmembers(instance):
            if name.startswith('_'):
                continue

            if isinstance(attr, MethodStub):
                setattr(header, name, attr.method(instance, name))
            elif isinstance(attr, State):
                setattr(header, name, type(attr)(instance_uri.append(name)))
            else:
                setattr(header, name, attr)

        form = {}
        for name, attr in inspect.getmembers(instance):
            if name.startswith('_'):
                continue
            elif name in parent_members:
                if attr is parent_members[name]:
                    continue
                elif hasattr(attr, "__code__") and hasattr(parent_members[name], "__code__"):
                    if attr.__code__ is parent_members[name].__code__:
                        continue

            if isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(header, name))
            else:
                form[name] = attr

        return form

    def __json__(cls):
        mro = cls.mro()
        if len(mro) < 2:
            raise ValueError("Tinychain class must extend a subclass of State")

        parent = mro[1]
        return {str(uri(parent)): to_json(form_of(cls))}


class MethodStub(object):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "cannot call a MethodStub; use tc.use(<class>) for callable method references")

    def method(self, header, name):
        return self.dtype(header, self.form, name)


class Method(object):
    __uri__ = uri(op.Op)

    def __init__(self, header, form, name):
        self.header = header
        self.form = form
        self.name = name

    def __json__(self):
        return {str(uri(self)): to_json(form_of(self))}

    def dtype(self):
        return self.__class__.__name__


class GetMethod(Method):
    __uri__ = uri(op.Get)

    def __call__(self, key=None):
        from .value import Nil
        rtype = inspect.signature(self.form).return_annotation
        rtype = resolve_class(self.form, rtype, Nil)
        return rtype(ref.Get(uri(self.header).append(self.name), key))

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) < 1 or len(parameters) > 3:
            raise ValueError(f"{self.dtype()} takes 1-3 arguments: (self, cxt, key)")

        args = [self.header]

        cxt = Context()
        if len(parameters) > 1:
            args.append(cxt)

        key_name = "key"
        if len(parameters) == 3:
            from .value import Value
            key_name, param = parameters[2]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

        cxt._return = self.form(*args) # populate the Context
        return (key_name, cxt)


class PutMethod(Method):
    __uri__ = uri(op.Put)

    def __call__(self, key, value):
        return ref.Put(uri(self.header).append(self.name), key, value)

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) not in [1, 2, 4]:
            raise ValueError(f"{self.dtype()} has one, two, or four arguments: "
                + "(self, cxt, key, value)")

        args = [self.header]

        cxt = Context()
        if len(parameters) > 1:
            args.append(cxt)

        key_name = "key"
        value_name = "value"
        if len(parameters) == 4:
            from .value import Value
            key_name, param = parameters[2]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

            value_name, param = parameters[3]
            dtype = resolve_class(self.form, param.annotation)
            args.append(dtype(URI(value_name)))

        cxt._return = self.form(*args)
        return (key_name, value_name, cxt)


class PostMethod(Method):
    __uri__ = uri(op.Post)

    def __call__(self, **params):
        from .value import Nil

        rtype = inspect.signature(self.form).return_annotation
        rtype = resolve_class(self.form, rtype, Nil)
        return rtype(ref.Post(uri(self.header).append(self.name), **params))

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) == 0:
            raise ValueError(f"{self.dtype()} has at least one argment: "
                + "(self, cxt, name1=val1, ...)")

        args = [self.header]

        cxt = Context()
        if len(parameters) > 1:
            args.append(cxt)

        kwargs = {}
        for name, param in parameters[2:]:
            dtype = resolve_class(self.form, param.annotation)
            kwargs[name] = dtype(URI(name))

        cxt._return = self.form(*args, **kwargs)
        return cxt


class DeleteMethod(Method):
    __uri__ = uri(op.Delete)

    def __call__(self, key=None):
        return ref.Delete(uri(self.header).append(self.name), key)

    def __form__(self):
        return GetMethod.__form__(self)


Method.Get = GetMethod
Method.Put = PutMethod
Method.Post = PostMethod
Method.Delete = DeleteMethod


class Op(object):
    __uri__ = uri(op.Op)

    def __init__(self, form):
        self.form = form

    def __json__(self):
        return {str(uri(self)): to_json(form_of(self))}

    def dtype(self):
        return self.__class__.__name__


class Get(Op):
    __uri__ = uri(op.Get)

    def __call__(self, key=None):
        return ref.Get(uri(self), key)

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) > 2:
            raise ValueError(f"{self.dtype()} takes 0-2 arguments: (cxt, key)")

        args = []

        cxt = Context()
        if len(parameters):
            args.append(cxt)

        key_name = "key"
        if len(parameters) == 2:
            from .value import Value
            key_name, param = parameters[1]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

        cxt._return = self.form(*args) # populate the Context
        return (key_name, cxt)

    def __ref__(self, name):
        return op.Get(URI(name))


class Put(Op):
    __uri__ = uri(op.Put)

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) not in [0, 1, 3]:
            raise ValueError(f"{self.dtype()} has 0, 1, or 3 arguments: (cxt, key, value)")

        args = [self.header]

        cxt = Context()
        if len(parameters):
            args.append(cxt)

        key_name = "key"
        value_name = "value"
        if len(parameters) == 3:
            from .value import Value
            key_name, param = parameters[1]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

            value_name, param = parameters[2]
            dtype = resolve_class(self.form, param.annotation)
            args.append(dtype(URI(value_name)))

        cxt._return = self.form(*args)
        return (key_name, value_name, cxt)

    def __ref__(self, name):
        return op.Put(URI(name))


class Post(Op):
    __uri__ = uri(op.Post)

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        args = []

        cxt = Context()
        if len(parameters):
            args.append(cxt)

        kwargs = {}
        for name, param in parameters[1:]:
            dtype = resolve_class(self.form, param.annotation)
            kwargs[name] = dtype(URI(name))

        cxt._return = self.form(*args, **kwargs)
        return cxt

    def __ref__(self, name):
        return op.Post(URI(name))


class Delete(Op):
    __uri__ = uri(op.Delete)

    def __form__(self):
        return Get.__form__(self)


    def __ref__(self, name):
        return op.Delete(URI(name))


def resolve_class(subject, annotation, default=State):
    if annotation == inspect.Parameter.empty:
        return default
    elif inspect.isclass(annotation):
        return annotation

    classpath = f"{subject.__module__}.{annotation}"
    resolved = locate(classpath)
    if resolved is None:
        raise ValueError(f"unable to resolve class {classpath}")
    else:
        return resolved

