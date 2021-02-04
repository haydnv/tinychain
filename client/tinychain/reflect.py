import inspect

from .state import Scalar, State, Value
from .util import *


class Header(object):
    pass


class Instance(object):
    def __init__(self, subject):
        self.subject = subject
        self.header = Header()

    def __form__(self):
        mro = self.subject.__class__.mro()
        parent_members = (
            {name for name, _ in inspect.getmembers(mro[1])}
            if len(mro) > 1 else set())

        for name, attr in inspect.getmembers(self.subject):
            if name.startswith('_') or name in parent_members:
                continue

            if isinstance(attr, MethodStub):
                setattr(self.header, name, MethodHeader(attr))
            elif isinstance(attr, State):
                setattr(self.header, name, type(attr)(URI(f"self/{name}")))
            else:
                raise AttributeError(f"invalid attribute {attr}")

        form = {}
        for name, attr in inspect.getmembers(self.subject):
            if name.startswith('_') or name in parent_members:
                continue

            if isinstance(attr, MethodStub):
                form[name] = to_json(attr(self.header))
            else:
                form[name] = attr

        return form


class MethodHeader(object):
    def __init__(self, stub):
        self.stub = stub

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


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


Method.Get = GetMethod


def num_args(sig):
    return len(sig.parameters)

def init_args(params, expect):
    args = []

    for param, dtype in zip(params, expect):
        name, param = param
        if param.annotation in {inspect.Parameter.empty, dtype}:
            args.append(dtype(URI(name)))
        elif issubclass(param.annotation, dtype):
            args.append(param.annotation(URI(name)))
        else:
            raise ValueError(f"{op} arg {name} should be of type {dtype}, not {param.annotation}")

    return args

