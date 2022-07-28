import dataclasses
import inspect

from ..json import to_json
from ..scalar import Value
from ..scalar import op, ref
from ..state import State
from ..uri import URI

from .functions import get_rtype, parse_args, resolve_class


EMPTY = inspect.Parameter.empty


class Method(object):
    __uri__ = URI(op.Op)

    @classmethod
    def expand(cls, header, form, name):
        yield name, cls(header, form, name)

    def __init__(self, header, form, name):
        if not inspect.isfunction(form):
            raise ValueError(f"reflection requires a Python method, not {form}")

        self.header = header
        self.form = form
        self.name = name
        self.sig = inspect.signature(self.form)

        if tuple(self.sig.parameters)[0] != "self":
            raise TypeError(f"method signature must begin with 'self' (found {tuple(self.sig.parameters.items())})")

    def __json__(self):
        return {str(self.__uri__): to_json(ref.form_of(self))}

    def __ref__(self):
        raise NotImplementedError

    def subject(self):
        return self.header.__uri__.append(self.name)


class Get(Method):
    __uri__ = URI(op.Get)

    def __init__(self, header, form, name):
        Method.__init__(self, header, form, name)
        self.rtype = get_rtype(self.form, State)

    def __call__(self, key=None):
        return self.rtype(form=ref.Get(self.subject(), key))

    def __form__(self):
        cxt, args = first_params(self)

        key_name = "key"
        if len(self.sig.parameters) == len(args):
            pass
        elif len(self.sig.parameters) - len(args) == 1:
            key_name = tuple(self.sig.parameters.keys())[-1]
            param = self.sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(form=URI(key_name)))
        else:
            raise ValueError(f"{self} should 0-3 parameters: (self, cxt, key)")

        if key_name in cxt:
            raise RuntimeError(f"namespace collision: {key_name} in {self.form}")

        cxt._return = self.form(*args)  # populate the Context

        return key_name, cxt

    def __ref__(self, name):
        sig = tuple(self.sig.parameters.items())
        assert 0 < len(sig) <= 3

        if len(sig) == 3 or (len(sig) == 2 and sig[1][0] not in ["cxt", "txn"]):
            ktype = resolve_class(self.form, sig[-1][1].annotation, Value)
        else:
            ktype = Value

        return op.Get[ktype, self.rtype](URI(name))


class Put(Method):
    __uri__ = URI(op.Put)

    def __init__(self, header, form, name):
        rtype = get_rtype(form, None)

        if not ref.is_none(rtype):
            raise ValueError(f"PUT method must return None, not f{rtype}")

        Method.__init__(self, header, form, name)

    def __call__(self, key, value):
        return ref.Put(self.subject(), key, value)

    def __form__(self):
        cxt, args = first_params(self)

        key_name = "key"
        value_name = "value"

        if len(self.sig.parameters) == len(args):
            pass
        elif len(self.sig.parameters) - len(args) == 1:
            name = tuple(self.sig.parameters.keys())[-1]
            param = self.sig.parameters[name]
            if name == key_name:
                dtype = resolve_class(self.form, param.annotation, Value)
                args.append(dtype(form=URI(key_name)))
            elif name == value_name:
                dtype = resolve_class(self.form, param.annotation, State)
                args.append(dtype(form=URI(value_name)))
            else:
                raise ValueError(f"a PUT method with three parameters requires an explicit 'key' or 'value', not '{name}'")
        elif len(self.sig.parameters) - len(args) == 2:
            key_name, value_name = tuple(self.sig.parameters.keys())[-2:]

            param = self.sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(form=URI(key_name)))

            param = self.sig.parameters[value_name]
            dtype = resolve_class(self.form, param.annotation, State)
            args.append(dtype(form=URI(value_name)))
        else:
            raise ValueError("a PUT method requires 0-4 parameters: (self, cxt, key, value), " +
                             f"not {tuple(self.sig.parameters.items())}")

        cxt._return = self.form(*args)

        for name in [key_name, value_name]:
            if name in cxt:
                raise RuntimeError(f"namespace collision: {name} in {self.form}")

        return key_name, value_name, cxt

    def __ref__(self, name):
        sig = tuple(self.sig.parameters.items())
        assert 0 < len(sig) <= 4

        if len(sig) >= 2 and sig[1][0] in ["cxt", "txn"]:
            sig = sig[2:]
        else:
            sig = sig[1:]

        if len(sig) == 1 and sig[0][0] == "key":
            ktype = resolve_class(self.form, sig[0][1].annotation, Value)
            vtype = State
        elif len(sig) == 1 and sig[0][0] == "value":
            ktype = Value
            vtype = resolve_class(self.form, sig[0][1].annotation, State)
        elif len(sig) == 2:
            ((_, k), (_, v)) = sig
            ktype = resolve_class(self.form, k.annotation, Value)
            vtype = resolve_class(self.form, v.annotation, State)
        elif not sig:
            ktype = Value
            vtype = State
        else:
            raise ValueError("a PUT method requires 0-4 parameters: (self, cxt, key, value), " +
                             f"not {tuple(self.sig.parameters.items())}")

        return op.Put[ktype, vtype](URI(name))


class Post(Method):
    __uri__ = URI(op.Post)

    def __init__(self, header, form, name):
        Method.__init__(self, header, form, name)
        self.rtype = get_rtype(self.form, State)

    def __call__(self, *args, **kwargs):
        sig = tuple(self.sig.parameters.items())
        rtype = self.rtype

        if not sig:
            raise TypeError(f"the method signature for {self} is missing the 'self' parameter")
        elif sig[0][0] != "self":
            raise TypeError(f"a method signature must begin with 'self', not '{sig[0][0]}'")

        if len(sig) > 1 and sig[1][0] in ["cxt", "txn"]:
            sig = sig[2:]
        else:
            sig = sig[1:]

        params = parse_args(sig, *args, **kwargs)
        return rtype(form=ref.Post(self.subject(), params))

    def __form__(self):
        cxt, args = first_params(self)

        kwargs = {}
        for name, param in tuple(self.sig.parameters.items())[len(args):]:
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

    def __ref__(self, name):
        fields = []

        sig = tuple(self.sig.parameters.items())
        sig = sig[2:] if len(sig) > 1 and sig[1][0] in ["cxt", "txn"] else sig[1:]
        for param_name, param in sig:
            dtype = resolve_class(self.subject(), param.annotation, State)
            if param.default is inspect.Parameter.empty:
                fields.append((param_name, dtype))
            else:
                fields.append((param_name, dtype, dataclasses.field(default=param.default)))

        sig = dataclasses.make_dataclass("Args", fields)
        return op.Post[sig, self.rtype](URI(name))


class Delete(Method):
    __uri__ = URI(op.Delete)

    def __init__(self, header, form, name):
        rtype = get_rtype(form, None)

        if not ref.is_none(rtype):
            raise ValueError(f"DELETE method must return None, not f{rtype}")

        Method.__init__(self, header, form, name)

    def __call__(self, key=None):
        return ref.Delete(self.subject(), key)

    def __form__(self):
        return Get.__form__(self)

    def __ref__(self, name):
        sig = tuple(self.sig.parameters.items())
        assert 0 < len(sig) <= 3

        if len(sig) == 3 or (len(sig) == 2 and sig[1][0] not in ["cxt", "txn"]):
            ktype = resolve_class(self.form, sig[-1][1].annotation, Value)
        else:
            ktype = Value

        return op.Delete[ktype](URI(name))


def _check_context_param(parameter):
    from ..context import Context

    _name, param = parameter
    if param.annotation == EMPTY or param.annotation == Context:
        pass
    else:
        raise ValueError(
            f"a method definition takes a transaction context as its second parameter, not {param.annotation}")


def first_params(method):
    from ..context import Context

    sig = inspect.signature(method.form)

    if not sig.parameters:
        raise ValueError(f"a method has at least one argument: (self, cxt, name1=val1, ...)")

    args = []

    param_names = tuple(sig.parameters.keys())

    if param_names[0] == "self":
        args.append(method.header)
    else:
        raise ValueError(f"the first argument to a method must be 'self', not {param_names[0]}")

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
