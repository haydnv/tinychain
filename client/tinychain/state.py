from . import reflect
from .util import *


# Base types (should not be instantiated directly

class State(object):
    __ref__ = URI("/state")

    def __init__(self, ref):
        if ref is None:
            raise ValueError("Null reference")

        self.__ref__ = ref

        reflect.gen_headers(self)

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


# Reference types

class Ref(object):
    __uri__ = uri(Scalar) + "/ref"


class IdRef(Ref):
    __uri__ = uri(Ref) + "/id"

    def __init__(self, subject):
        self.subject = subject

    def __json__(self):
        return {str(self): []}

    def __str__(self):
        return f"${self.subject}"


class After(Ref):
    __uri__ = uri(Ref) + "/after"

    def __init__(self, when, then):
        self.when = when
        self.then = then

    def __json__(self):
        return {str(uri(self)): to_json([self.when, self.then])}


class If(Ref):
    __uri__ = uri(Ref) + "/if"

    def __init__(self, cond, then, or_else):
        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.then, self.or_else])}


class OpRef(Ref):
    __uri__ = uri(Ref) + "/op"

    def __init__(self, subject, args):
        if isinstance(subject, State):
            self.subject = uri(subject)
        else:
            self.subject = subject

        self.args = args

    def __json__(self):
        return {str(self.subject): to_json(self.args)}


class GetOpRef(OpRef):
    __uri__ = uri(OpRef) + "/get"

    def __init__(self, subject, key=None):
        OpRef.__init__(self, subject, (key,))


class PutOpRef(OpRef):
    __uri__ = uri(OpRef) + "/put"

    def __init__(self, subject, key, value):
        OpRef.__init__(self, subject, (key, value))


class PostOpRef(OpRef):
    __uri__ = uri(OpRef) + "/post"

    def __init__(self, subject, **kwargs):
        OpRef.__init__(self, subject, kwargs)


class DeleteOpRef(OpRef):
    __uri__ = uri(OpRef) + "/delete"

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


