import inspect

from .util import *


# Base types (should not be initialized directly)

class State(object):
    PATH = "/state"

    def __init__(self, state_spec):
        if isinstance(state_spec, self.__class__):
            self.__spec__ = spec(state_spec)
        else:
            self.__spec__ = state_spec

    def __json__(self):
        return {self.PATH: [to_json(spec(self))]}

    def __ref__(self, name=None):
        if isinstance(spec(self), Ref):
            if name is None:
                return self
        elif name is None:
            raise ValueError(f"Reference to {self} requires a name")
        else:
            return type(self)(IdRef(name))


class Scalar(State):
    PATH = State.PATH + "/scalar"

    def __init__(self, spec):
        State.__init__(self, spec)

    def __json__(self):
        return to_json({self.PATH: spec(self)})

    def __setattr__(self, attr, value):
        State.__setattr__(self, attr, value)


class Ref(Scalar):
    PATH = Scalar.PATH + "/ref"


# Reference types

class IdRef(Ref):
    PATH = Ref.PATH + "/id"

    def __init__(self, subject):
        if isinstance(subject, Ref):
            self.__spec__ = spec(subject)
        elif isinstance(subject, State) and isinstance(spec(subject), Ref):
            self.__spec__ = spec(spec(subject))
        elif hasattr(subject, "__ref__"):
            raise ValueError(
                f"Ref to {subject} needs a name (hint: try calling ref(spec, 'name')")
        else:
            self.__spec__ = subject

        if str(spec(self)).startswith("$"):
            raise ValueError(f"Ref name cannot start with '$': {self.name}")

    def __json__(self):
        return to_json({str(self): []})

    def __str__(self):
        return f"${spec(self)}"


class OpRef(Ref):
    PATH = Ref.PATH + "/op"


class GetOpRef(OpRef):
    PATH = OpRef.PATH + "/get"

    def __init__(self, link, key=None):
        link = Link(link)
        OpRef.__init__(self, (link, key))

    def __json__(self):
        (link, key) = spec(self)
        return to_json({str(link): [key]})


class PutOpRef(OpRef):
    PATH = OpRef.PATH + "/put"

    def __init__(self, link, key, value):
        link = Link(link)
        OpRef.__init__(self, (link, key, value))

    def __json__(self):
        subject, key, value = spec(self)
        return {str(subject): [to_json(key), to_json(value)]}


class PostOpRef(OpRef):
    PATH = OpRef.PATH + "/post"

    def __init__(self, link, **params):
        link = Link(link)
        OpRef.__init__(self, (link, params))

    def __json__(self):
        subject, params = spec(self)
        return {str(subject): to_json(params)}


class DeleteOpRef(OpRef):
    PATH = OpRef.PATH + "/delete"

    def __init__(self, link, key=None):
        link = Link(link)
        OpRef.__init__(self, (link, key))

    def __json__(self):
        (link, key) = spec(self)
        return to_json({str(link): [key]})


OpRef.Get = GetOpRef
OpRef.Put = PutOpRef
OpRef.Post = PostOpRef
OpRef.Delete = PostOpRef


class After(Ref):
    PATH = Ref.PATH + "/after"

    def __init__(self, when, then):
        Ref.__init__(self, (when, then))


class If(Ref):
    PATH = Ref.PATH + "/if"

    def __init__(self, cond, then, or_else):
        Ref.__init__(self, (cond, then, or_else))


# Compound types

class Map(State):
    PATH = State.PATH + "/map"


class Tuple(State):
    PATH = State.PATH + "/tuple"


# Op types

class Op(Scalar):
    PATH = Scalar.PATH + "/op"

    def __init__(self, spec):
        if isinstance(spec, Ref) or inspect.isfunction(spec):
            Scalar.__init__(self, spec)
        else:
            raise ValueError("Op spec must be a callable function")


class GetOp(Op):
    PATH = Op.PATH + "/get"

    def __call__(self, key=None):
        return OpRef.Get(IdRef(self), key)

    def __form__(self):
        args = []
        context = Context()

        params = list(inspect.signature(spec(self)).parameters.items())
        if params:
            first_annotation = params[0][1].annotation
            if first_annotation == inspect.Parameter.empty or first_annotation == Context:
                args.append(context)
            else:
                raise ValueError(
                    "The first argument to an Op definition is a transaction context "
                     + f"(`tc.Context`), not {first_annotation}")

        if len(params) == 2:
            key_name = params[1][0]
            key_annotation = params[1][1].annotation
            if key_annotation == inspect.Parameter.empty:
                args.append(IdRef(key_name))
            elif issubclass(key_annotation, Value):
                args.append(key_annotation(IdRef(key_name)))
            else:
                raise ValueError(f"GET Op key ({key_name}) must be a Value")

        if len(params) > 2:
            raise ValueError("GET Op takes two optional parameters, a Context and a Value")

        result = spec(self)(*args) # populate the Context
        if isinstance(result, State):
            context._result = result
        else:
            raise ValueError(f"Op return value must be a Tinychain state, not {result}")

        return context

    def __json__(self):
        params = list(inspect.signature(spec(self)).parameters.items())
        if len(params) > 2:
            raise ValueError(f"GET Op takes two optional parameters, a Context and a Value")
        elif len(params) == 2:
            key_name = params[1][0]
        else:
            key_name = "key"

        return to_json({self.PATH: [key_name, form_of(self)]})



class PutOp(Op):
    PATH = Op.PATH + "/put"

    def __form__(self):
        raise NotImplemented


class PostOp(Op):
    PATH = Op.PATH + "/post"

    def __form__(self):
        form = spec(self)

        if not inspect.isfunction(form):
            return to_json(form)

        args = []
        context = Context()

        params = list(inspect.signature(form).parameters.items())
        if params:
            first_annotation = params[0][1].annotation
            if first_annotation == inspect.Parameter.empty or first_annotation == Context:
                args.append(context)
            else:
                raise ValueError(
                    "The first argument to an Op definition is a transaction context "
                     + f"(`tc.Context`), not {first_annotation}")

            for name, param in params[1:]:
                if param.annotation == inspect.Parameter.empty:
                    args.append(IdRef(name))
                else:
                    args.append(param.annotation(IdRef(name)))

        result = spec(self)(*args) # populate the Context
        if isinstance(result, _State) or isinstance(result, Ref):
            context._result = result
        else:
            raise ValueError(f"Op return value must be a Tinychain state, not {result}")

        return context


class DeleteOp(Op):
    PATH = Op.PATH + "/delete"

    def __form__(self):
        raise NotImplemented


Op.Get = GetOp
Op.Put = PutOp
Op.Post = PostOp
Op.Delete = DeleteOp


# Value types

class Value(Scalar):
    PATH = Scalar.PATH + "/value"


class Nil(Value):
    PATH = Value.PATH + "/none"


class Number(Value):
    PATH = Value.PATH + "/number"

    def __json__(self):
        return to_json(spec(self))

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def add(self, other):
        raise NotImplementedError

    def mul(self, other):
        raise NotImplementedError


class Bool(Number):
    PATH = Number.PATH + "/bool"


class Complex(Number):
    PATH = Number.PATH + "/complex"

    def __json__(self):
        return to_json({self.PATH: spec(self)})


class C32(Complex):
    PATH = Complex.PATH + "/32"


class C64(Complex):
    PATH = Complex.PATH + "/64"


class Float(Number):
    PATH = Number.PATH + "/float"


class F32(Float):
    PATH = Float.PATH + "/32"


class F64(Float):
    PATH = Float.PATH + "/64"


class Int(Number):
    PATH = Number.PATH + "/int"


class I16(Int):
    PATH = Int.PATH + "/16"


class I32(Int):
    PATH = Int.PATH + "/32"


class I64(Int):
    PATH = Int.PATH + "/64"


class UInt(Number):
    PATH = Number.PATH + "/uint"


class U8(UInt):
    PATH = UInt.PATH + "/64"


class U16(UInt):
    PATH = UInt.PATH + "/64"


class U32(UInt):
    PATH = UInt.PATH + "/64"


class U64(UInt):
    PATH = UInt.PATH + "/64"


class String(Value):
    PATH = Value.PATH + "/string"

    def __json__(self):
        return str(spec(self))


class Link(Value):
    PATH = Value.PATH + "/link"

    def __json__(self):
        return {str(self): []}

    def __str__(self):
        return str(spec(self))

