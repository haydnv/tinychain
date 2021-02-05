import inspect

from . import error
from .state import OpRef, Scalar, State
from .util import *
from .value import Nil, Value


class Class(object):
    def __init__(self, cls):
        class Instance(cls):
            def __init__(self, ref):
                cls.__init__(self, ref)

                for name, attr in inspect.getmembers(self):
                    if name.startswith('_'):
                        continue

                    if isinstance(attr, MethodStub):
                        setattr(self, name, method_header(self, name, attr))

        self.cls = cls
        self.instance = Instance

    def __call__(self, ref):
        return self.instance(ref)

    def __form__(self):
        mro = self.cls.mro()
        parent_members = (
            {name for name, _ in inspect.getmembers(mro[1])}
            if len(mro) > 1 else set())


        class Header(self.cls):
            pass

        header = Header(URI("self"))
        instance = self.cls(URI("self"))

        for name, attr in inspect.getmembers(instance):
            if name.startswith('_') or name in parent_members:
                continue

            if isinstance(attr, MethodStub):
                setattr(header, name, method_header(instance, name, attr))
            elif isinstance(attr, State):
                setattr(header, name, type(attr)(URI(f"self/{name}")))
            else:
                setattr(header, name, attr)

        form = {}
        for name, attr in inspect.getmembers(instance):
            if name.startswith('_') or name in parent_members:
                continue

            if isinstance(attr, MethodStub):
                form[name] = to_json(attr(header))
            else:
                form[name] = attr

        return form

    def __json__(self):
        return {str(uri(self.cls)): to_json(form_of(self))}


def method_header(subject, name, stub):
    return_type = inspect.signature(stub.form).return_annotation
    if return_type == inspect.Signature.empty:
        return_type = Nil

    method_uri = uri(subject).append(name)

    if stub.dtype == Method.Get:
        def get_header(key=None):
            return return_type(OpRef.Get(method_uri, key))

        return get_header
    elif stub.dtype == Method.Post:
        def post_header(**kwargs):
            return return_type(OpRef.Post(method_uri, kwargs))

        return post_header
    else:
        raise error.MethodNotAllowed(f"Unrecognized method type: {stub.dtype}")


class MethodStub(object):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def __call__(self, header):
        return self.dtype(header, self.form)


class Method(object):
    __ref__ = uri(Scalar) + "/op"

    def __init__(self, header, form):
        self.header = header
        self.form = form

    def __json__(self):
        return {str(uri(self)): to_json(form_of(self))}


class GetMethod(Method):
    __ref__ = uri(Method) + "/get"

    def __form__(self):
        sig = inspect.signature(self.form)

        if num_args(sig) < 1 or num_args(sig) > 3:
            raise ValueError("GET method takes 1-3 arguments: (self, context, key)")

        args = [self.header]

        cxt = Context()
        if num_args(sig) > 1:
            args.append(cxt)

        key_name = "key"
        if num_args(sig) == 3:
            key_name, param = sig.parameters[2]
            if param.annotation in {inspect.Parameter.empty, Value}:
                args.append(Value(URI(key_name)))
            elif issubclass(param.annotation, Value):
                args.append(param.annotation(URI(key_name)))

        cxt._return = self.form(*args) # populate the Context
        return (key_name, cxt)


class PostMethod(Method):
    __def__ = uri(Method) + "/get"

    def __form__(self):
        sig = inspect.signature(self.form)

        if num_args(sig) == 0:
            raise ValueError("POST method has at least one argment: (self, name1=val1, ...)")

        args = [self.header]
        kwargs = {}

        cxt = Context()
        if num_args(sig) > 1:
            args.append(cxt)

        for name, param in list(sig.parameters.items())[2:]:
            dtype = State if param.annotation == inspect.Parameter.empty else param.annotation
            kwargs[name] = dtype(URI(name))

        cxt._return = self.form(*args, **kwargs)
        return cxt


Method.Get = GetMethod
Method.Post = PostMethod


def num_args(sig):
    return len(sig.parameters)

