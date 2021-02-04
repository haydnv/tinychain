from .util import *


# Base types (should not be instantiated directly

class State(object):
    __ref__ = URI("/state")

    def __init__(self, ref):
        self.__ref__ = ref

    def __json__(self):
        return to_json(ref(self))

    @classmethod
    def get(cls, key=None):
        return cls(OpRef.Get(uri(cls), key))

# Scalar types

class Scalar(State):
    __ref__ = uri(State) + "/scalar"


# Scalar value types

class Value(Scalar):
    __ref__ = uri(Scalar) + "/value"


# Numeric types


class Number(Value):
    __ref__ = uri(Value) + "/number"

# Reference types

class Ref(object):
    __ref__ = uri(Scalar) + "/ref"


class OpRef(Ref):
    __ref__ = uri(Ref) + "/op"

    def __init__(self, subject, args):
        self.subject = uri(subject)
        self.args = args

    def __json__(self):
        return {str(self.subject): to_json(self.args)}


class GetOpRef(OpRef):
    __ref__ = uri(OpRef) + "/get"

    def __init__(self, subject, key=None):
        OpRef.__init__(self, subject, (key,))


OpRef.Get = GetOpRef


# User-defined object types


class Meta(type):
    __ref__ = uri(State) + "/object/class"

    def __form__(cls):
        return reflect.Instance(cls())

    def __json__(cls):
        return {uri(cls): form_of(cls)}


class Method(Scalar):
    __ref__ = uri(Scalar) + "/op"


class GetMethod(Method):
    __ref__ = uri(Method) + "/get"


Method.Get = GetMethod

