from .util import *


# Base types (should not be instantiated directly

class State(object):
    __ref__ = URI("/state")

    def __init__(self, ref):
        assert ref is not None
        self.__ref__ = ref

    def __json__(self):
        return to_json(ref(self))
    
    @classmethod
    def init(cls, key=None):
        if isinstance(ref(cls), URI):
            return cls(OpRef.Get(uri(cls), key))
        else:
            return cls(OpRef.Get(cls, key))


# Scalar types

class Scalar(State):
    __ref__ = uri(State) + "/scalar"


# Scalar value types

class Value(Scalar):
    __ref__ = uri(Scalar) + "/value"


class Nil(Value):
    __ref__ = uri(Value) + "/none"


# Reference types

class Ref(object):
    __ref__ = uri(Scalar) + "/ref"


class OpRef(Ref):
    __ref__ = uri(Ref) + "/op"

    def __init__(self, subject, args):
        if isinstance(subject, State):
            self.subject = uri(subject)
        else:
            self.subject = subject

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


class Class(State):
    __ref__ = uri(State) + "/object/class"


