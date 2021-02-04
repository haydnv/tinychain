from .util import *


# Base types (should not be instantiated directly

class State(object):
    __uri__ = URI("/state")

    def __init__(self, spec=None, uri=None):
        if spec:
            self.__spec__ = spec

        if uri:
            self.__uri__ = uri

    def __json__(self):
        return to_json({str(uri(self)): form_of(self)})

    def __ref__(self, uri):
        copy = type(self)(spec=self.__spec__)
        copy.__uri__ = uri
        return copy


# Scalar types

class Scalar(State):
    __uri__ = uri(State) + "/scalar"

    def __form__(self):
        if hasattr(self, "__spec__"):
            return self.__spec__
        else:
            return {str(uri(self)): []}


# Op types

class Op(Scalar):
    __uri__ = uri(Scalar) + "/op"


class GetOp(Op):
    __uri__ = uri(Op) + "/get"


Op.Get = GetOp

# Scalar value types

class Value(Scalar):
    __uri__ = uri(Scalar) + "/value"


# Number types

class Number(Value):
    __uri__ = uri(Value) + "/number"

    def __mul__(self, other):
        return self.mul(other)

    def mul(self, other):
        return Number(spec=[other], uri=uri(self).append("mul"))


# User-defined object types

class Class(State):
    __uri__ = uri(State) + "/object/class"

    def __init__(self, class_def):
        class Instance(class_def):
            pass

        State.__init__(self, spec=Instance)

    def __call__(self, spec=None, uri=None):
        return self.__spec__(spec, uri)

    def __form__(self):
        mro = self.__spec__.mro()
        parent_members = (
            {name for name, _ in inspect.getmembers(mro[2])}
            if len(mro) > 2 else set())

        form = {}

        for name, attr in inspect.getmembers(self.__spec__):
            if name.startswith('_') or name in parent_members:
                continue

            if isinstance(attr, MethodStub):
                form[name] = attr(self(uri=URI("$self")))
            else:
                form[name] = attr

        return {str(uri(self.__spec__)): form}


class Method(Op):
    pass


class GetMethod(Method):
    __uri__ = uri(Op.Get)

    def __form__(self):
        (subject, form) = self.__spec__

        args = []
        context = Context()

        params = list(inspect.signature(form).parameters.items())
        assert params[0][0] == "self"
        if len(params) < 1 or len(params) > 3:
            raise ValueError("GET method takes 1-3 parameters")

        first_annotation = params[0][1].annotation
        if first_annotation in {inspect.Parameter.empty, type(subject)}:
            args.append(subject)
        else:
            raise ValueError("The first argument to Method.Get is an instance of"
                             + f"{subject.__class__}, not {first_annotation}")

        if len(params) > 1:
            second_annotation = params[1][1].annotation
            if second_annotation == inspect.Parameter.empty or second_annotation == Context:
                args.append(context)
            else:
                raise ValueError("The second argument to Method.Get is a transaction"
                                 + f"Context, not {second_annotation}")

        if len(params) == 3:
            key_name, param = params[2]
            if param.annotation == inspect.Parameter.empty:
                args.append(Value(uri=URI(f"${key_name}")))
            elif issubclass(param.annotation, Value):
                args.append(param.annotation(uri=URI(f"${key_name}")))
            else:
                raise ValueError("GET Method key must be a Tinychain Value")

        result = form(*args) # populate the Context
        if isinstance(result, State):
            context._result = result
        else:
            raise ValueError(f"Op return value must be a Tinychain state, not {result}")

        return context

    def __json__(self):
        return to_json(form_of(self))

Method.Get = GetMethod


class MethodStub(object):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def __call__(self, subject):
        return self.dtype(spec=(subject, self.form))

