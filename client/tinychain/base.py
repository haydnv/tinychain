"""The base class of :class:`State` and :class:`Interface`"""

import inspect
import typing

from .reflect import MethodStub

BUILTINS = set(["shape"])


class _Base(object):
    def __init__(self):
        attr_names = set(dir(self)) - set(dir(object)) - BUILTINS

        for name in attr_names:
            if name.startswith('_'):
                continue

            attr = getattr(self, name)

            if isinstance(attr, MethodStub):
                method = attr.method(self, name)
                setattr(self, name, method)

    def _get(self, name, key=None, rtype=None):
        from .scalar.ref import Get, MethodSubject

        subject = MethodSubject(self, name)
        op_ref = Get(subject, key)
        rtype = _resolve_rtype(rtype)
        return rtype(form=op_ref)

    def _put(self, name, key=None, value=None):
        from .scalar.ref import MethodSubject, Put
        from .scalar.value import Nil

        subject = MethodSubject(self, name)
        return Nil(Put(subject, key, value))

    def _post(self, name, params, rtype):
        from .scalar.ref import MethodSubject, Post

        subject = MethodSubject(self, name)
        op_ref = Post(subject, params)
        rtype = _resolve_rtype(rtype)
        return rtype(form=op_ref)

    def _delete(self, name, key=None):
        from .scalar.ref import Delete, MethodSubject
        from .scalar.value import Nil

        subject = MethodSubject(self, name)
        return Nil(Delete(subject, key))


def _resolve_rtype(rtype, default=None):
    from .state import State
    if default is None:
        default = State

    if typing.get_origin(rtype) is tuple:
        from .generic import Tuple
        return Tuple.expect(rtype)
    elif typing.get_origin(rtype) is dict:
        from .generic import Map
        return Map.expect(rtype)
    elif inspect.isclass(rtype) and issubclass(rtype, State):
        return rtype
    else:
        return default
