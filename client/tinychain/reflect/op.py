import inspect

from tinychain import op, ref
from tinychain.state import State
from tinychain.util import form_of, requires, to_json, uri, Context, URI
from tinychain.value import Nil, Value

from . import _get_rtype, resolve_class


EMPTY = inspect.Parameter.empty


class Op(object):
    __uri__ = uri(op.Op)

    def __init__(self, form):
        self.form = form

    def __deps__(self):
        params = inspect.signature(self.form).parameters
        provided = set(URI(param) for param in params)
        return requires(form_of(self)) - provided

    def __json__(self):
        return {str(uri(self)): to_json(form_of(self))}

    def dtype(self):
        return self.__class__.__name__


class Get(Op):
    __uri__ = uri(op.Get)

    def __call__(self, key=None):
        return ref.Get(self, key)

    def __form__(self):
        cxt, args = _maybe_first_arg(self)

        sig = inspect.signature(self.form)
        key_name = "key"
        if len(sig.parameters) > len(args):
            key_name = list(sig.parameters.keys())[len(args)]
            param = sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

        cxt._return = self.form(*args)  # populate the Context
        return key_name, cxt

    def __ref__(self, name):
        return op.Get(URI(name))

    def __repr__(self):
        return f"GET Op with form {to_json(self)}"


class Put(Op):
    __uri__ = uri(op.Put)

    def __init__(self, form):
        rtype = _get_rtype(form, None)
        if rtype not in [None, Nil]:
            raise ValueError(f"{self.dtype()} can only return None, not f{rtype}")

        self.rtype = rtype
        Op.__init__(self, form)

    def __call__(self, key=None, value=None):
        return ref.Put(self, key, value)

    def __form__(self):
        cxt, args = _maybe_first_arg(self)

        sig = inspect.signature(self.form)
        key_name = "key"
        value_name = "value"

        if len(sig.parameters) == len(args):
            pass
        elif len(sig.parameters) - len(args) == 1:
            param_name = list(sig.parameters.keys())[-1]
            param = sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            if param_name in ["key", "value"]:
                args.append(dtype(URI(param_name)))
            else:
                raise ValueError(f"{self.dtype()} argument {param_name} is ambiguous--use 'key' or 'value' instead")
        elif len(sig.parameters) - len(args) == 2:
            param_names = list(sig.parameters.keys())
            key_name = param_names[-2]
            param = sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

            value_name = param_names[-1]
            param = sig.parameters[value_name]
            dtype = resolve_class(self.form, param.annotation, State)
            args.append(dtype(URI(value_name)))
        else:
            param_names = list(sig.parameters.keys())
            raise ValueError(
                f"{self.dtype()} accepts a maximum of three arguments: (cxt, key, value) (found {param_names})")

        cxt._return = self.form(*args)
        return key_name, value_name, cxt

    def __ref__(self, name):
        return op.Put(URI(name))

    def __repr__(self):
        return f"PUT Op with form {self.form}"


class Post(Op):
    __uri__ = uri(op.Post)

    def __init__(self, form):
        self.rtype = _get_rtype(form, State)
        Op.__init__(self, form)

    def __call__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("POST Op takes one arg (a Map) or kwargs, but not both")

        if not args:
            args = kwargs

        return ref.Get(self, args)

    def __form__(self):
        cxt, args = _maybe_first_arg(self)

        sig = inspect.signature(self.form)
        kwargs = {}
        for name, param in list(sig.parameters.items())[len(args):]:
            dtype = resolve_class(self.form, param.annotation, State)
            kwargs[name] = dtype(URI(name))

        cxt._return = self.form(*args, **kwargs)
        return cxt

    def __ref__(self, name):
        return op.Post(URI(name))

    def __repr__(self):
        return f"POST Op with form {self.form}"


class Delete(Op):
    __uri__ = uri(op.Delete)

    def __init__(self, form):
        rtype = _get_rtype(form, None)
        if rtype not in [None, Nil]:
            raise ValueError(f"Delete op can only return None, not f{rtype}")

        self.rtype = rtype
        Op.__init__(self, form)

    def __call__(self, key=None):
        return ref.Get(self, key)

    def __form__(self):
        return Get.__form__(self)

    def __ref__(self, name):
        return op.Delete(URI(name))

    def __repr__(self):
        return f"DELETE Op with form {self.form}"


def _maybe_first_arg(op):
    sig = inspect.signature(op.form)
    param_names = list(sig.parameters.keys())

    cxt = Context()
    args = []

    if sig.parameters:
        first_param = param_names[0]
        if first_param == "cxt" or first_param == "txn":
            param = sig.parameters[first_param]
            if param.annotation == EMPTY or param.annotation is Context:
                args.append(cxt)
            else:
                raise ValueError(f"{param} must be a {Context}, not {param.annotation}")

    return cxt, args
