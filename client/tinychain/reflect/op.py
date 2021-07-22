import inspect

from tinychain import op, ref
from tinychain.state import State
from tinychain.util import form_of, to_json, uri, Context, URI
from tinychain.value import Nil, Value

from . import _get_rtype, resolve_class


EMPTY = inspect.Parameter.empty


class Op(object):
    __uri__ = uri(op.Op)

    def __init__(self, form):
        self.form = form

    def __json__(self):
        return {str(uri(self)): to_json(form_of(self))}

    def dtype(self):
        return self.__class__.__name__


class Get(Op):
    __uri__ = uri(op.Get)

    def __call__(self, key=None):
        return ref.Get(uri(self), key)

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) > 2:
            raise ValueError(f"{self.dtype()} takes 0-2 arguments: (cxt, key)")

        args = []

        cxt = Context()
        if len(parameters):
            args.append(cxt)

        key_name = "key"
        if len(parameters) == 2:
            key_name, param = parameters[1]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

        cxt._return = self.form(*args)  # populate the Context
        return key_name, cxt

    def __ref__(self, name):
        return op.Get(URI(name))


class Put(Op):
    __uri__ = uri(op.Put)

    def __init__(self, form):
        rtype = _get_rtype(form, None)
        if rtype not in [None, Nil]:
            raise ValueError(f"Put op can only return None, not f{rtype}")

        self.rtype = rtype
        Op.__init__(self, form)

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) not in [0, 1, 3]:
            raise ValueError(f"{self.dtype()} has 0, 1, or 3 arguments: (cxt, key, value)--found {parameters}")

        args = []

        cxt = Context()
        if len(parameters):
            args.append(cxt)

        key_name = "key"
        value_name = "value"
        if len(parameters) == 3:
            key_name, param = parameters[1]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

            value_name, param = parameters[2]
            dtype = resolve_class(self.form, param.annotation, State)
            args.append(dtype(URI(value_name)))

        cxt._return = self.form(*args)
        return key_name, value_name, cxt

    def __ref__(self, name):
        return op.Put(URI(name))


class Post(Op):
    __uri__ = uri(op.Post)

    def __init__(self, form):
        self.rtype = _get_rtype(form, State)
        Op.__init__(self, form)

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        args = []

        cxt = Context()
        if parameters:
            args.append(cxt)

        kwargs = {}
        for name, param in parameters[1:]:
            dtype = resolve_class(self.form, param.annotation, State)
            kwargs[name] = dtype(URI(name))

        cxt._return = self.form(*args, **kwargs)
        return cxt

    def __ref__(self, name):
        return op.Post(URI(name))


class Delete(Op):
    __uri__ = uri(op.Delete)

    def __init__(self, form):
        rtype = _get_rtype(form, None)
        if rtype not in [None, Nil]:
            raise ValueError(f"Delete op can only return None, not f{rtype}")

        self.rtype = rtype
        Op.__init__(self, form)

    def __form__(self):
        return Get.__form__(self)

    def __ref__(self, name):
        return op.Delete(URI(name))
