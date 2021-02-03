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

    def get(self, path="/", key=None):
        return MethodRef.Get(self, path)(key)

    def put(self, path, key, value):
        return MethodRef.Put(self, path)(key, value)

    def post(self, path="/", **params):
        return MethodRef.Post(self, path)(**params)

    def delete(self, path="/", key=None):
        return MethodRef.Delete(self, path)(key)


class Scalar(State):
    PATH = State.PATH + "/scalar"

    def __init__(self, spec):
        State.__init__(self, spec)
        self.__lock__ = True

    def __json__(self):
        return to_json({self.PATH: spec(self)})

    def __setattr__(self, attr, value):
        if locked(self):
            raise RuntimeError("a Scalar is immutable at runtime")

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

    def __json__(self):
        subject, key, value = spec(self)
        return {str(subject): [to_json(key), to_json(value)]}


class PostOpRef(OpRef):
    PATH = OpRef.PATH + "/post"

    def __json__(self):
        subject, params = spec(self)
        return {str(subject): to_json(params)}


class DeleteOpRef(OpRef):
    PATH = OpRef.PATH + "/delete"


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

    def __init__(self, spec):
        self.add = MethodRef.Get(spec, "/add")

        Value.__init__(self, spec)

    def __json__(self):
        return to_json(spec(self))

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)


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


# User-defined class & instance types

class Class(State):
    PATH = State.PATH + "/class"

    def __init__(self, spec):
        if not inspect.isclass(spec):
            raise ValueError("Class spec must be a class definition")

        class Instance(spec):
            PATH = spec.PATH

            def __init__(self, instance_spec):
                self.__spec__ = instance_spec

                raise NotImplementedError

            def __getattr__(self, name):
                form = self.__form__()
                if name in form:
                    return ref(form[name], name)

        self.__spec__ = Instance

    def __call__(self, instance_spec):
        return spec(self)(instance_spec)

    def __getattr__(self, name):
        return getattr(spec(self), name)

    def __json__(self):
        extends = spec(self).PATH
        proto = form_of(self(IdRef("self")))

        return {Class.PATH: to_json({extends: proto})}


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
        self.subject = subject
        Op.__init__(self, spec)


class GetMethod(Method):
    PATH = Op.Get.PATH

    def __init__(self, subject, spec):
        Method.__init__(self, subject, spec)

    def __form__(self):
        args = []
        context = Context()

        params = list(inspect.signature(spec(self)).parameters.items())
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

        result = spec(self)(*args) # populate the Context
        if isinstance(result, State) or isinstance(result, Ref):
            context._value = result
        else:
            raise ValueError(f"Op return value must be a Tinychain state, not {result}")

        return context

    def __json__(self):
        params = list(inspect.signature(spec(self)).parameters.items())
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


class PutCall(MethodCall):
    PATH = OpRef.Put.PATH

    def __init__(self, method, key, value):
        self.method = method
        self.key = key
        self.value = value

    def __json__(self):
        return {str(self.method): to_json([self.key, self.value])}


class PostCall(MethodCall):
    PATH = OpRef.Post.PATH

    def __init__(self, method, **params):
        self.method = method
        self.params = params

    def __json__(self):
        return {str(self.method): to_json(self.params)}


class DeleteCall(MethodCall):
    PATH = OpRef.Delete.PATH

    def __init__(self, method, key=None):
        self.method = method
        self.key = key

    def __json__(self):
        return {self.PATH: to_json([self.method.subject, self.method.path, self.key])}


MethodCall.Get = GetCall
MethodCall.Put = PutCall
MethodCall.Post = PostCall
MethodCall.Delete = DeleteCall


class MethodRef:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Cannot instantiate MethodRef directly; try Method.<access method>")

    class Get(AttrRef):
        def __call__(self, key=None):
            return MethodCall.Get(self, key)

    class Post(AttrRef):
        def __call__(self, **params):
            return MethodCall.Post(self, **params)

    class Put(AttrRef):
        def __call__(self, key, value):
            return MethodCall.Put(self, key, value)

    class Delete(AttrRef):
        def __call__(self, key=None):
            return MethodCall.Delete(self, key)



class MethodStub(object):
    def __init__(self, dtype, method):
        self.dtype = dtype
        self.method = method

    def __call__(self, subject):
        return self.dtype(subject, self.method)

