import inspect

from .util import *


# Base types (should not be instantiated directly)

class State(object):
    __uri__ = URI("/state")

    def __init__(self, uri=None):
        if uri is None:
            self.__uri__ = uri(self.__class__)
        else:
            self.__uri__ = uri

    def init(self, form):
        self.__form__ = lambda: form
        return self

    def __json__(self):
        return to_json({str(self.__class__): form_of(self)})


class Scalar(State):
    __uri__ = uri(State) + "/scalar"


# Op types

class Op(Scalar):
    __uri__ = uri(Scalar) + "/op"


class GetOp(Op):
    __uri__ = uri(Op) + "/get"

    def init(self, form):
        if not inspect.isfunction(form):
            raise ValueError(f"Op form must be a callable function, not {form}")

        args = []
        context = Context()

        params = list(inspect.signature(form).parameters.items())

        raise NotImplementedError


Op.Get = GetOp


# Scalar value types

class Value(Scalar):
    __uri__ = uri(Scalar) + "/value"


class Number(Value):
    __uri__ = uri(Value) + "/number"

    def __mul__(self, other):
        subject = uri(self).append("mul")
        return OpRef.Get(uri(self).append("mul"), other)


# Reference types

class OpRef(object):
    __uri__ = uri(Scalar) + "/ref/op"

    def __init__(self, subject, args):
        self.subject = subject
        self.args = args

    def __form__(self):
        return to_json(self)

    def __json__(self):
        return to_json({str(self.subject): self.args})


class GetOpRef(OpRef):
    __uri__ = uri(OpRef) + "/get"

    def __init__(self, subject, key=None):
        OpRef.__init__(self, subject, (key,))


OpRef.Get = GetOpRef


# Types to support user-defined objects

def get_method(form):
    return MethodStub(Method.Get, form)


class Class(State):
    __uri__ = uri(State) + "/object/class"

    def __init__(self, cls):
        if not inspect.isclass(cls):
            raise ValueError(f"Class requires a Python class definition (not {cls})")

        class Instance(cls):
            def __form__(self):
                mro = self.__class__.mro()
                parent_members = (
                    {name for name, _ in inspect.getmembers(mro[2])}
                    if len(mro) > 2
                    else set())

                form = {}
                for name, attr in inspect.getmembers(self):
                    if name.startswith('_') or name in parent_members:
                        continue

                    if isinstance(attr, MethodStub):
                        form[name] = attr(self, name)
                    else:
                        form[name] = attr

                return form

        self.instance = Instance

        State.__init__(self, uri(cls))

    def __call__(self, uri=None):
        return self.instance(uri)

    def __json__(self):
        extends = uri(self)
        instance = self(URI("$self"))
        return to_json({
            str(uri(Class)): {
                str(extends): form_of(instance)
            }
        })


class Method(Op):
    pass


class GetMethod(Method, Op.Get):
    def init(self, subject, form):
        args = []
        context = Context()

        params = list(inspect.signature(form).parameters.items())

        assert params[0][0] == "self"
        if len(params) > 3:
            raise ValueError("a GET method takes no more than three parameters")

        args.append(subject)

        if len(params) >= 2:
            first_annotation = params[1][1].annotation
            if first_annotation == inspect.Parameter.empty or first_annotation == Context:
                args.append(context)
            else:
                raise ValueError(
                    "The first argument to an Op definition is a transaction context "
                     + f"(`tc.Context`), not {first_annotation}")

        if len(params) == 3:
            key_name = params[1][0]
            key_type = params[1][1].annotation
            if key_type == inspect.Parameter.empty:
                args.append(Value(URI(key_name)))
            else:
                args.append(key_type(URI(key_name)))

        for name, param in params[1:]:
            if param.annotation == inspect.Parameter.empty:
                args.append(Ref(name))
            else:
                args.append(param.annotation(Ref(name)))

        context._result = form(*args)

        self.__form__ = lambda: context

    def __json__(self):
        return to_json(form_of(self))


Method.Get = GetMethod


class MethodStub(object):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def __call__(self, subject, name):
        method = self.dtype(uri(subject).append(name))
        method.init(subject, self.form)
        return method


# Convenience methods

def const(value, dtype=Scalar):
    c = dtype().init(value)
    return c

