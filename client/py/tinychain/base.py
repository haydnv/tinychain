"""The base class of :class:`State` and :class:`Interface`"""

import inspect
import typing

from .reflect.stub import MethodStub
from .uri import URI

BUILTINS = set(["shape"])
INHERITED = set(dir(object))


class _Base(object):
    def __init__(self):
        attr_names = set(name for name in dir(self) if not name.startswith("__"))
        attr_names = attr_names - INHERITED - BUILTINS

        for name in attr_names:
            if name.startswith('_'):
                continue

            attr = getattr(self, name)

            if isinstance(attr, MethodStub):
                for method_name, method in attr.expand(self, name):
                    setattr(self, method_name, method)

    def _get(self, name, key=None, rtype=None):
        from .scalar.ref import Get

        subject = URI(self, name) if name else self
        op_ref = Get(subject, key)
        rtype = _resolve_rtype(rtype)
        return rtype(form=op_ref)

    def _put(self, name, key=None, value=None):
        from .scalar.ref import Put
        from .scalar.value import Nil

        subject = URI(self, name) if name else self
        return Nil(Put(subject, key, value))

    def _post(self, name, params, rtype):
        from .scalar.ref import Post

        subject = URI(self, name) if name else self
        op_ref = Post(subject, params)
        rtype = _resolve_rtype(rtype)
        return rtype(form=op_ref)

    def _delete(self, name, key=None):
        from .scalar.ref import Delete
        from .scalar.value import Nil

        subject = URI(self, name) if name else self
        return Nil(Delete(subject, key))


def _resolve_rtype(rtype, default=None):
    from .state import State
    if default is None:
        default = State

    if typing.get_origin(rtype) is tuple:
        from .generic import Tuple
        return Tuple[rtype]
    elif typing.get_origin(rtype) is dict:
        from .generic import Map
        return Map[rtype]
    elif inspect.isclass(rtype) and issubclass(rtype, State):
        return rtype
    else:
        return default
