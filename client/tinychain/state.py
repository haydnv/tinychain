from . import reflect
from .ref import OpRef
from .util import *


# Base type (should not be instantiated directly

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
        return self.__class__(URI(name))

    def __repr__(self):
        return f"{self.__class__.__name__}(form_of(self))"

    def dtype(self):
        return Class(OpRef.get(uri(self) + "/class"))


class Map(State):
    __uri__ = uri(State) + "/map"

    def __json__(self):
        return to_json(form_of(self))

class Tuple(State):
    __uri__ = uri(State) + "/tuple"

    def __json__(self):
        return to_json(form_of(self))


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

    def __call__(self, key=None):
        return OpRef.Get(self, key)


class PutOp(Op):
    __uri__ = uri(Op) + "/put"

    def __call__(self, key=None, value=None):
        return OpRef.Put(self, key, value)


class PostOp(Op):
    __uri__ = uri(Op) + "/post"

    def __call__(self, **params):
        return OpRef.Post(self, **params)


class DeleteOp(Op):
    __uri__ = uri(Op) + "/delete"

    def __call__(self, key=None):
        return OpRef.Delete(self, key)


Op.Get = GetOp
Op.Put = PutOp
Op.Post = PostOp
Op.Delete = DeleteOp


# User-defined object types

class Class(State):
    __uri__ = uri(State) + "/object/class"


