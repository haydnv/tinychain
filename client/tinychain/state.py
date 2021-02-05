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

    def __repr__(self):
        return f"{self.__class__.__name__}({ref(self)})"


# Scalar types

class Scalar(State):
    __ref__ = uri(State) + "/scalar"

    def _get(self, name, dtype):
        setattr(self, name, lambda key: dtype(OpRef.Get(uri(self).append(name), key)))


# Reference types

class Ref(object):
    __ref__ = uri(Scalar) + "/ref"


class If(Ref):
    __ref__ = uri(Ref) + "/if"

    def __init__(self, cond, then, or_else):
        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.then, self.or_else])}


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


class PutOpRef(OpRef):
    __ref__ = uri(OpRef) + "/put"

    def __init__(self, subject, key, value):
        OpRef.__init__(self, subject, (key, value))


class PostOpRef(OpRef):
    __ref__ = uri(OpRef) + "/post"

    def __init__(self, subject, **kwargs):
        OpRef.__init__(self, subject, kwargs)


class DeleteOpRef(OpRef):
    __ref__ = uri(OpRef) + "/delete"

    def __init__(self, subject, key=None):
        OpRef.__init__(self, subject, key)

    def __json__(self):
        return {str(uri(DeleteOpRef)): to_json([self.subject, self.args])}


OpRef.Get = GetOpRef
OpRef.Put = PutOpRef
OpRef.Post = PostOpRef
OpRef.Delete = DeleteOpRef


# User-defined object types


class Class(State):
    __ref__ = uri(State) + "/object/class"


