import inspect

from .util import Context, form_of, to_json


# Base types (should not be initialized directly)


class State(object):
    PATH = "/state"

    def __init__(self, spec):
        if isinstance(spec, self.__class__):
            self.spec = spec.spec
        else:
            self.spec = spec

    def __json__(self):
        return {self.PATH: [to_json(self.spec)]}


class Scalar(State):
    PATH = State.PATH + "/scalar"

    def __json__(self):
        return to_json({self.PATH: self.spec})


class Ref(Scalar):
    PATH = Scalar.PATH + "/ref"


# Reference types

class IdRef(Ref):
    PATH = Ref.PATH + "/id"

    def __json__(self):
        return to_json({f"${self.spec}": []})


class OpRef(Ref):
    PATH = Ref.PATH + "/op"


class GetOpRef(OpRef):
    PATH = OpRef.PATH + "/get"

    def __init__(self, link, key=None):
        link = Link(link)
        OpRef.__init__(self, (link, key))

    def __json__(self):
        (link, key) = self.spec
        return to_json({str(link): [key]})


OpRef.Get = GetOpRef


# Scalar tuple

class Tuple(Scalar):
    PATH = Scalar.PATH + "/tuple"


# Object types

class Map(Scalar):
    PATH = Scalar.PATH + "/map"

    def __json__(self):
        json = {}
        for name in self.spec:
            json[name] = to_json(self.spec[name])

        return json


# Op types

class Op(Scalar):
    PATH = Scalar.PATH + "/op"

    def __init__(self, spec):
        if not inspect.isfunction(spec):
            raise ValueError("Op spec must be a callable function")

        Scalar.__init__(self, spec)


class GetOp(Op):
    PATH = Op.PATH + "/get"

    def __form__(self):
        args = []
        context = Context()

        params = list(inspect.signature(self.spec).parameters.items())
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

        result = self.spec(*args) # populate the Context
        if isinstance(result, State):
            context._result = result
        else:
            raise ValueError(f"Op return value must be a Tinychain state, not {result}")

        return context

    def __json__(self):
        params = list(inspect.signature(self.spec).parameters.items())
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
        if not inspect.isfunction(self.spec):
            return to_json(self.spec)

        args = []
        context = Context()

        params = list(inspect.signature(self.spec).parameters.items())
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

        result = self.spec(*args) # populate the Context
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
        return to_json(self.spec)


class Bool(Number):
    PATH = Number.PATH + "/bool"


class Complex(Number):
    PATH = Number.PATH + "/complex"

    def __json__(self):
        return to_json({self.PATH: self.spec})


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
        return str(self.spec)


class Link(Value):
    PATH = Value.PATH + "/link"

    def __json__(self):
        return {str(self): []}

    def __str__(self):
        return self.spec

