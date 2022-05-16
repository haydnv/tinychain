import inspect

from ..scalar import Value
from ..scalar import op, ref
from ..state import State
from ..uri import uri, URI
from ..context import to_json, Context

from . import _get_rtype, parse_args, resolve_class


EMPTY = inspect.Parameter.empty


class Method(object):
    __uri__ = uri(op.Op)

    def __init__(self, header, form, name):
        if not inspect.isfunction(form):
            raise ValueError(f"reflection requires a Python method, not {form}")

        self.header = header
        self.form = form
        self.name = name

    def __eq__(self, other):
        if isinstance(other, Method):
            return self.form == other.form

        return False

    def __json__(self):
        return {str(uri(self)): to_json(ref.form_of(self))}

    def dtype(self):
        return self.__class__.__name__

    def subject(self):
        if isinstance(self.header, State):
            return ref.MethodSubject(self.header, self.name)
        else:
            return uri(self.header).append(self.name)


class Get(Method):
    __uri__ = uri(op.Get)

    def __init__(self, header, form, name):
        Method.__init__(self, header, form, name)
        self.rtype = _get_rtype(self.form, State)

    def __call__(self, key=None):
        return self.rtype(form=ref.Get(self.subject(), key))

    def __args__(self):
        _, cxt = ref.form_of(self)
        return [self.subject(), cxt]

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
            args.append(dtype(form=URI(key_name)))
        else:
            raise ValueError(f"{self.dtype()} takes 0-3 parameters: (self, cxt, key)")

        if key_name in cxt:
            raise RuntimeError(f"namespace collision: {key_name} in {self.form}")

        cxt._return = self.form(*args)  # populate the Context

        return key_name, cxt


class Put(Method):
    __uri__ = uri(op.Put)

    def __init__(self, header, form, name):
        rtype = _get_rtype(form, None)

        if not ref.is_none(rtype):
            raise ValueError(f"PUT method must return None, not f{rtype}")

        Method.__init__(self, header, form, name)

    def __call__(self, key, value):
        return ref.Put(self.subject(), key, value)

    def __args__(self):
        _, _, cxt = ref.form_of(self)
        return [self.subject(), cxt]

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
                args.append(dtype(form=URI(key_name)))
            elif name == value_name:
                dtype = resolve_class(self.form, param.annotation, State)
                args.append(dtype(form=URI(value_name)))
            else:
                raise ValueError(f"{self.dtype()} with three parameters requires 'key' or 'value', not '{name}'")
        elif len(sig.parameters) - len(args) == 2:
            key_name, value_name = list(sig.parameters.keys())[-2:]

            param = sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(form=URI(key_name)))

            param = sig.parameters[value_name]
            dtype = resolve_class(self.form, param.annotation, State)
            args.append(dtype(form=URI(value_name)))
        else:
            raise ValueError(f"{self.dtype()} requires 0-4 parameters: (self, cxt, key, value)")

        cxt._return = self.form(*args)

        for name in [key_name, value_name]:
            if name in cxt:
                raise RuntimeError(f"namespace collision: {name} in {self.form}")

        return key_name, value_name, cxt


class Post(Method):
    __uri__ = uri(op.Post)

    def __init__(self, header, form, name):
        Method.__init__(self, header, form, name)
        self.rtype = _get_rtype(self.form, State)

    def __call__(self, *args, **kwargs):
        sig = list(inspect.signature(self.form).parameters.items())
        rtype = self.rtype

        if not sig:
            raise TypeError(f"POST method signature for {self} is missing the 'self' parameter")
        elif sig[0][0] != "self":
            raise TypeError(f"POST method signature must begin with 'self', not '{sig[0][0]}'")

        if len(sig) > 1 and sig[1][0] in ["cxt", "txn"]:
            sig = sig[2:]
        else:
            sig = sig[1:]

        params = parse_args(sig, *args, **kwargs)
        return rtype(form=ref.Post(self.subject(), params))

    def __args__(self):
        _, cxt = ref.form_of(self)
        return [self.subject(), cxt]

    def __form__(self):
        cxt, args = _first_params(self)

        sig = inspect.signature(self.form)
        kwargs = {}
        for name in list(sig.parameters.keys())[len(args):]:
            param = sig.parameters[name]

            dtype = State
            if param.default is inspect.Parameter.empty:
                if param.annotation:
                    dtype = param.annotation
            elif isinstance(param.default, State):
                dtype = type(param.default)

            dtype = resolve_class(self.form, dtype, State)
            kwargs[name] = dtype(form=URI(name))

        cxt._return = self.form(*args, **kwargs)

        for name in kwargs.keys():
            if name in cxt:
                raise RuntimeError(f"namespace collision: {name} in {self.form}")

        return cxt


class Delete(Method):
    __uri__ = uri(op.Delete)

    def __init__(self, header, form, name):
        rtype = _get_rtype(form, None)

        if not ref.is_none(rtype):
            raise ValueError(f"DELETE method must return None, not f{rtype}")

        Method.__init__(self, header, form, name)

    def __call__(self, key=None):
        return ref.Delete(self.subject(), key)

    def __args__(self):
        _, cxt = ref.form_of(self)
        return [self.subject(), cxt]

    def __form__(self):
        return Get.__form__(self)


class Operator(Post):
    def __call__(self, *args, **kwargs):
        result = self.form(self.header, *args, **kwargs)

        if self.rtype:
            return self.rtype(form=result)
        else:
            return result


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
