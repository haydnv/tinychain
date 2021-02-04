import inspect

from .state import State, Value
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
                sig = inspect.signature(attr.form)

                if sig.return_annotation == inspect.Signature.empty:
                    setattr(self.header, name, None)
                else:
                    setattr(self.header, name, sig.return_annotation(URI(name)))

            elif isinstance(attr, State):
                setattr(self.header, name, type(attr)(URI(f"self/{name}")))
            else:
                setattr(self.header, name, attr)

        form = {}
        for name, attr in inspect.getmembers(self.subject):
            if name.startswith('_') or name in parent_members:
                continue

            if isinstance(attr, MethodStub):
                form[name] = form_of(attr(self.header))
            else:
                form[name] = attr

        return form


class MethodStub(object):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def __call__(self, header):
        return self.dtype(header, self.form)


class Method(object):
    def __init__(self, header, form):
        self.header = header
        self.form = form


class GetMethod(Method):
    def __form__(self):
        sig = inspect.signature(self.form)
        args = [self.header]

        expect = [Context, Value]
        args.extend(init_args(list(sig.parameters.items())[1:], expect))

        return self.form(*args)


Method.Get = GetMethod


def num_args(sig):
    return len(inspect.signature(op).parameters)

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

