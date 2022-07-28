import dataclasses
import inspect
import logging

from ..json import to_json
from ..scalar.value import Value
from ..scalar import op, ref
from ..state import State
from ..uri import URI

from .functions import get_rtype, resolve_class


EMPTY = inspect.Parameter.empty


class Op(object):
    __uri__ = URI(op.Op)

    def __init__(self, form):
        if not inspect.isfunction(form):
            raise ValueError(f"reflection requires a Python callable, not {form}")

        self.form = form
        self.sig = inspect.signature(self.form)

    def __json__(self):
        return {str(URI(self)): to_json(ref.form_of(self))}

    def __ref__(self):
        raise RuntimeError("cannot reference a reflected Op; use Get, Put, Post, or Delete instead")

    def dtype(self):
        return self.__class__.__name__


class Get(Op):
    __uri__ = URI(op.Get)

    def __init__(self, form):
        self.rtype = get_rtype(form, State)
        Op.__init__(self, form)

    def __form__(self):
        cxt, args = maybe_first_arg(self.form)

        key_name = "key"
        if len(self.sig.parameters) > len(args):
            key_name = list(self.sig.parameters.keys())[len(args)]
            param = self.sig.parameters[key_name]
            ktype = resolve_class(self.form, param.annotation, Value)
            args.append(ktype(form=URI(key_name)))

        cxt._return = self.form(*args)  # populate the Context

        validate(cxt, self.sig.parameters)

        return key_name, cxt

    def __ref__(self, name):
        sig = tuple(self.sig.parameters.items())
        assert len(sig) <= 2

        if sig and sig[0][0] in ["cxt", "txn"]:
            sig = sig[1:]

        if len(sig) == 1:
            ktype = resolve_class(self.form, sig[0][1].annotation, Value)
        else:
            assert not sig
            ktype = Value

        return op.Get[ktype, self.rtype](form=URI(name))

    def __repr__(self):
        return f"GET Op with form {to_json(self)}"


class Put(Op):
    __uri__ = URI(op.Put)

    def __call__(self, key=None, value=None):
        return ref.Put(self, key, value)

    def __form__(self):
        cxt, args = maybe_first_arg(self.form)

        key_name = "key"
        value_name = "value"

        if len(self.sig.parameters) == len(args):
            pass
        elif len(self.sig.parameters) - len(args) == 1:
            param_name = list(self.sig.parameters.keys())[-1]
            param = self.sig.parameters[param_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            if param_name in set(["key", "value"]):
                args.append(dtype(form=URI(param_name)))
            else:
                raise ValueError(f"{self.dtype()} argument {param_name} is ambiguous--use 'key' or 'value' instead")
        elif len(self.sig.parameters) - len(args) == 2:
            param_names = list(self.sig.parameters.keys())
            key_name = param_names[-2]
            param = self.sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(form=URI(key_name)))

            value_name = param_names[-1]
            param = self.sig.parameters[value_name]
            dtype = resolve_class(self.form, param.annotation, State)
            args.append(dtype(URI(value_name)))
        else:
            param_names = list(self.sig.parameters.keys())
            raise ValueError(
                f"{self.dtype()} accepts a maximum of three arguments: (cxt, key, value) (found {param_names})")

        cxt._return = self.form(*args)

        validate(cxt, self.sig.parameters)

        return key_name, value_name, cxt

    def __ref__(self, name):
        sig = tuple(self.sig.parameters.items())
        assert len(sig) < 3

        if sig and sig[0][0] in ["cxt", "txn"]:
            sig = sig[1:]

        if len(sig) == 2:
            (_kn, k), (_vn, v) = sig
            ktype = resolve_class(self.form, k.annotation, Value)
            vtype = resolve_class(self.form, v.annotation, State)
        elif len(sig) == 1 and sig[0][0] == "key":
            ktype = resolve_class(self.form, sig[0][1].annotation, Value)
            vtype = State
        elif len(sig) == 1 and sig[0][0] == "value":
            ktype = Value
            vtype = resolve_class(self.form, sig[0][1].annotation, State)
        elif not sig:
            ktype = Value
            vtype = State
        else:
            raise ValueError(f"invalid signature for PUT Op: {tuple(self.sig.parameters.items())}")

        return op.Put[ktype, vtype](form=URI(name))

    def __repr__(self):
        return f"PUT Op with form {self.form}"


class Post(Op):
    __uri__ = URI(op.Post)

    def __init__(self, form):
        self.rtype = get_rtype(form, State)
        Op.__init__(self, form)

    def __form__(self):
        cxt, args = maybe_first_arg(self.form)

        kwargs = {}
        for name, param in tuple(self.sig.parameters.items())[len(args):]:
            dtype = resolve_class(self.form, param.annotation, State)
            kwargs[name] = dtype(form=URI(name))

        cxt._return = self.form(*args, **kwargs)

        validate(cxt, self.sig.parameters)

        return cxt

    def __ref__(self, name):
        sig = list(self.sig.parameters.items())

        if sig and sig[0][0] in ["cxt", "txn"]:
            sig = sig[1:]

        fields = []
        for param_name, param in sig:
            dtype = resolve_class(self.form, param.annotation, State)
            if param.default is inspect.Parameter.empty:
                fields.append((param_name, dtype))
            else:
                fields.append((param_name, dtype, dataclasses.field(default=param.default)))

        sig = dataclasses.make_dataclass("Args", fields)
        return op.Post[sig, self.rtype](form=URI(name))

    def __repr__(self):
        return f"POST Op with form {self.form}"


class Delete(Op):
    __uri__ = URI(op.Delete)

    def __form__(self):
        return Get.__form__(self)

    def __ref__(self, name):
        sig = tuple(self.sig.parameters.items())
        assert len(sig) <= 2

        if sig and sig[0][0] in ["cxt", "txn"]:
            sig = sig[1:]

        if sig:
            ktype = resolve_class(self.form, sig[0][1].annotation, Value)
        else:
            ktype = Value

        return op.Delete[ktype](form=URI(name))

    def __repr__(self):
        return f"DELETE Op with form {self.form}"


def maybe_first_arg(form):
    from ..context import Context

    sig = inspect.signature(form)
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

    validate(cxt, sig.parameters)

    return cxt, args


def validate(cxt, provided):
    defined = set(provided)
    for name in provided:
        if name in cxt:
            raise RuntimeError(f"namespace collision: {name} in {cxt}")

    for name in cxt:
        def validate_ref(ref):
            if not hasattr(ref, "__uri__") and not isinstance(ref, URI):
                return

            ref = ref if isinstance(ref, URI) else URI(ref)
            if ref.id() is not None and ref.id() not in defined:
                logging.info(f"{cxt} depends on undefined state {ref.id()}--is it part of a Closure?")

        form = getattr(cxt, name)
        while hasattr(form, "__form__"):
            form = ref.form_of(form)

        if isinstance(form, URI):
            validate_ref(form)

        if isinstance(form, ref.Op):
            validate_ref(form.subject)

            if isinstance(form.args, (list, tuple)):
                for arg in form.args:
                    validate_ref(arg)
            elif isinstance(form.args, dict):
                for arg_name in form.args:
                    validate_ref(form.args[arg_name])
            else:
                validate_ref(form.args)

        defined.add(name)
