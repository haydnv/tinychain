import inspect

from tinychain import op, ref
from tinychain.state import State
from tinychain.util import form_of, requires, to_json, uri, Context, URI
from tinychain.value import Nil, Value

from . import _get_rtype, is_none, resolve_class


EMPTY = inspect.Parameter.empty


class Method(object):
    __uri__ = uri(op.Op)

    def __init__(self, header, form, name):
        self.header = header
        self.form = form
        self.name = name

    def __deps__(self):
        return requires(form_of(self.form))

    def __json__(self):
        return {str(uri(self)): to_json(form_of(self))}

    def dtype(self):
        return self.__class__.__name__


class Get(Method):
    __uri__ = uri(op.Get)

    def __init__(self, header, form, name):
        self.rtype = _get_rtype(form, State)
        Method.__init__(self, header, form, name)

    def __call__(self, key=None):
        return self.rtype(ref.Get(uri(self.header).append(self.name), key))

    def __form__(self):
        cxt, args = _first_params(self)

        sig = inspect.signature(self.form)
        key_name = "key"
        if len(sig.parameters) == len(args):
            pass
        elif len(sig.parameters) - len(args) == 1:
            key_name = list(sig.parameters.keys())[-1]
            param = sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))
        else:
            raise ValueError(f"{self.dtype()} takes 0-3 parameters: (self, cxt, key)")

        cxt._return = self.form(*args)  # populate the Context
        return key_name, cxt


class Put(Method):
    __uri__ = uri(op.Put)

    def __init__(self, header, form, name):
        rtype = _get_rtype(form, None)

        if not is_none(rtype):
            raise ValueError(f"Put method must return None, not f{rtype}")

        Method.__init__(self, header, form, name)

    def __call__(self, key, value):
        return ref.Put(uri(self.header).append(self.name), key, value)

    def __form__(self):
        cxt, args = _first_params(self)

        sig = inspect.signature(self.form)

        key_name = "key"
        value_name = "value"

        if len(sig.parameters) == len(args):
            pass
        elif len(sig.parameters) - len(args) == 1:
            name = list(sig.parameters.keys())[-1]
            param = sig.parameters[name]
            if name == key_name:
                dtype = resolve_class(self.form, param.annotation, Value)
                args.append(dtype(URI(key_name)))
            elif name == value_name:
                dtype = resolve_class(self.form, param.annotation, State)
                args.append(dtype(URI(value_name)))
            else:
                raise ValueError(f"{self.dtype()} with three parameters requires 'key' or 'value', not '{name}'")
        elif len(sig.parameters) - len(args) == 2:
            key_name, value_name = list(sig.parameters.keys())[-2:]

            param = sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

            param = sig.parameters[value_name]
            dtype = resolve_class(self.form, param.annotation, State)
            args.append(dtype(URI(value_name)))
        else:
            raise ValueError(f"{self.dtype()} requires 0-4 parameters: (self, cxt, key, value)")

        cxt._return = self.form(*args)
        return key_name, value_name, cxt


class Post(Method):
    __uri__ = uri(op.Post)

    def __init__(self, header, form, name):
        self.rtype = _get_rtype(form, State)
        Method.__init__(self, header, form, name)

    def __call__(self, params):
        rtype = inspect.signature(self.form).return_annotation
        rtype = resolve_class(self.form, rtype, Nil)
        return rtype(ref.Post(uri(self.header).append(self.name), params))

    def __form__(self):
        cxt, args = _first_params(self)

        sig = inspect.signature(self.form)
        kwargs = {}
        for name in list(sig.parameters.keys())[len(args):]:
            param = sig.parameters[name]
            dtype = resolve_class(self.form, param.annotation, State)
            kwargs[name] = dtype(URI(name))

        cxt._return = self.form(*args, **kwargs)
        return cxt


class Delete(Method):
    __uri__ = uri(op.Delete)

    def __init__(self, header, form, name):
        rtype = _get_rtype(form, None)

        if not is_none(rtype):
            raise ValueError(f"Delete method must return None, not f{rtype}")

        Method.__init__(self, header, form, name)

    def __call__(self, key=None):
        return ref.Delete(uri(self.header).append(self.name), key)

    def __form__(self):
        return Get.__form__(self)


def _check_context_param(parameter):
    _name, param = parameter
    if param.annotation == EMPTY or param.annotation == Context:
        pass
    else:
        raise ValueError(
            f"a method definition takes a transaction context as its second parameter, not {param.annotation}")


def _first_params(method):
    sig = inspect.signature(method.form)

    if not sig.parameters:
        raise ValueError(f"{method.dtype()} has at least one argument: (self, cxt, name1=val1, ...)")

    args = []

    param_names = list(sig.parameters.keys())

    if param_names[0] == "self":
        args.append(method.header)
    else:
        raise ValueError(f"first argument to {method.dtype()} must be 'self', not {param_names[0]}")

    cxt = Context()

    if len(args) == len(sig.parameters):
        return cxt, args

    if param_names[1] in ["cxt", "txn"]:
        param = sig.parameters[param_names[1]]
        if param.annotation in [EMPTY, Context]:
            args.append(cxt)
        else:
            raise ValueError(f"type of {param_names[1]} must be {Context}, not {param.annotation}")

    return cxt, args
