import inspect

from .util import *


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

    def __ref__(self, name=None):
        if isinstance(self.spec, Ref):
            if name is None:
                return self
            else:
                raise ValueError(f"Cannot rename {self}")
        elif name is None:
            raise ValueError(f"Reference to {self} requires a name")
        else:
            return type(self)(IdRef(name))

    def get(self, path="/", key=None):
        return MethodRef.Get(self, path)(key)


class Scalar(State):
    PATH = State.PATH + "/scalar"

    def __json__(self):
        return to_json({self.PATH: self.spec})


class Ref(Scalar):
    PATH = Scalar.PATH + "/ref"


# Reference types

class IdRef(Ref):
    PATH = Ref.PATH + "/id"

    def __init__(self, subject):
        if isinstance(subject, Ref):
            self.spec = subject.spec
        elif isinstance(subject, State) and isinstance(subject.spec, Ref):
            self.spec = subject.spec.spec
        elif hasattr(subject, "__ref__"):
            raise ValueError(
                f"Ref to {subject} needs a name (hint: try calling ref(spec, 'name')")
        else:
            self.spec = subject

        if str(self.spec).startswith("$"):
            raise ValueError(f"Ref name cannot start with '$': {self.name}")

    def __json__(self):
        return to_json({str(self): []})

    def __str__(self):
        return f"${self.spec}"


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


class Map(Scalar):
    PATH = Scalar.PATH + "/map"

    def __json__(self):
        json = {}
        for name in self.spec:
            json[name] = to_json(self.spec[name])

        return json


class Tuple(Scalar):
    PATH = Scalar.PATH + "/tuple"


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

    def __mul__(self, other):
        return self.mul(other)

    def mul(self, other):
        return Number(self.get("/mul", other))


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
        return str(self.spec)


# User-defined class & instance types

class Class(State):
    PATH = State.PATH + "/class"

    def __init__(self, spec):
        instance_class = self

        if not inspect.isclass(spec):
            raise ValueError("expected a class definition")

        class Instance(spec):
            def __init__(self, instance_spec):
                super().__init__(instance_spec)

                for name, method in inspect.getmembers(self):
                    if name.startswith('_'):
                        continue

                    if isinstance(method, MethodStub):
                        if isinstance(self.spec, Ref):
                            setattr(self, name, ref(method(self), name))
                        else:
                            setattr(self, name, ref(method(ref(self, "self")), name))

        self.spec = Instance

    def __call__(self, spec):
        return self.spec(spec)

    def __getattr__(self, name):
        return getattr(self.spec, name)

    def __json__(self):
        mro = self.spec.mro()
        if len(mro) < 3:
            raise ValueError("Tinychain Class must extend a native class (e.g. tc.State)")

        parent = mro[2]
        extends = parent.PATH
        proto = {}
        parent_members = set(name for name, _ in inspect.getmembers(parent))
        subject = self.spec(IdRef("self"))

        for name, method in inspect.getmembers(self.spec):
            if name.startswith('_') or name in parent_members:
                continue

            if isinstance(method, MethodStub):
                method = method(subject)
                proto[name] = to_json(method)
            else:
                raise ValueError(f"{name} is not a Method")

        return to_json({Class.PATH: {extends: proto}})


class AttrRef(Ref):
    def __init__(self, subject, path):
        self.subject = IdRef(subject)
        self.path = Link(path)

    def __json__(self):
        return {str(self): []}

    def __str__(self):
        if str(self.path) == "/":
            return str(self.subject)
        else:
            return f"{self.subject}{self.path}"


# Method definition & call utilities

class Method(Op):
    def __init__(self, subject, spec):
        Op.__init__(self, spec)
        self.subject = subject


class GetMethod(Method):
    PATH = Op.Get.PATH

    def __init__(self, subject, spec):
        Method.__init__(self, subject, spec)

    def __form__(self):
        args = []
        context = Context()

        params = list(inspect.signature(self.spec).parameters.items())
        if params:
            if len(params) > 3:
                raise ValueError("GET Method takes only one value parameter")

            if params:
                first_annotation = params[0][1].annotation
                if first_annotation in {inspect.Parameter.empty, type(self.subject)}:
                    args.append(self.subject)
                else:
                    raise ValueError("The first argument to Method.Get is an instance of"
                                     + f"{self.subject.__class__}, not {first_annotation}")

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
                    args.append(Ref(key_name))
                elif issubclass(param.annotation, Value):
                    args.append(param.annotation(Ref(key_name)))
                else:
                    raise ValueError("GET Method key must be a Tinychain Value")

        result = self.spec(*args) # populate the Context
        if isinstance(result, State) or isinstance(result, Ref):
            context._value = result
        else:
            raise ValueError(f"Op return value must be a Tinychain state, not {result}")

        return context

    def __json__(self):
        params = list(inspect.signature(self.spec).parameters.items())
        key_name = params[2][0] if len(params) == 3 else "key"
        return {self.PATH: [key_name, to_json(form_of(self))]}

    def __ref__(self, name):
        return MethodRef.Get(self.subject, f"/{name}")


Method.Get = GetMethod


class MethodCall(Ref):
    PATH = OpRef.PATH


class GetCall(MethodCall):
    PATH = OpRef.Get.PATH

    def __init__(self, method, key=None):
        self.method = method
        self.key = key

    def __json__(self):
        return {str(self.method): [to_json(self.key)]}


MethodCall.Get = GetCall


class MethodRef:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Cannot instantiate MethodRef directly; try Method.<access method>")

    class Get(AttrRef):
        def __call__(self, key=None):
            return MethodCall.Get(self, key)



class MethodStub(object):
    def __init__(self, dtype, method):
        self.dtype = dtype
        self.method = method

    def __call__(self, subject):
        return self.dtype(subject, self.method)

