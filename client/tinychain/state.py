from . import reflect
from .util import *


# Reference types

class Ref(object):
    __uri__ = URI("/state/scalar/ref")


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
        if isinstance(subject, Ref):
            self.subject = subject
        else:
            self.subject = uri(subject)

        self.args = args

    def __json__(self):
        return {str(self.subject): to_json(self.args)}


class GetOpRef(OpRef):
    __uri__ = uri(OpRef) + "/get"

    def __init__(self, subject, key=None):
        if str(uri(subject)).startswith("/state/scalar"):
            OpRef.__init__(self, subject, key)
        else:
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


# Base types (should not be instantiated directly

class State(object):
    __uri__ = URI("/state")

    def __init__(self, form):
        self.__form__ = form

        if isinstance(form, URI):
            self.__uri__ = form

        reflect.gen_headers(self)

    def __json__(self):
        return {str(uri(self)): [to_json(form_of(self))]}

    def __ref__(self, name):
        return self.__class__(IdRef(name))

    def __repr__(self):
        return f"{self.__class__.__name__}(form_of(self))"

    def dtype(self):
        return Class(OpRef.get(uri(self) + "/class"))


# Scalar types

class Scalar(State):
    __uri__ = uri(State) + "/scalar"

    def __json__(self):
        return to_json(form_of(self))

# User-defined Ops

class Op(Scalar):
    __uri__ = uri(Scalar) + "/op"


class GetOp(Op):
    __uri__ = uri(Op) + "/get"


class PutOp(Op):
    __uri__ = uri(Op) + "/put"


class PostOp(Op):
    __uri__ = uri(Op) + "/post"


class DeleteOp(Op):
    __uri__ = uri(Op) + "/delete"


Op.Get = GetOp
Op.Put = PutOp
Op.Post = PostOp
Op.Delete = DeleteOp


# User-defined object types

class Class(State):
    __uri__ = uri(State) + "/object/class"


