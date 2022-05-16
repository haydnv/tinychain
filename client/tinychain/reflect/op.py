import inspect
import logging

from ..scalar.value import Nil, Value
from ..scalar import op, ref
from ..state import State
from ..uri import uri, URI
from ..context import to_json, Context

from . import _get_rtype, parse_args, resolve_class


EMPTY = inspect.Parameter.empty


class Op(object):
    __uri__ = uri(op.Op)

    def __init__(self, form):
        if not inspect.isfunction(form):
            raise ValueError(f"reflection requires a Python callable, not {form}")

        self.form = form

    def __json__(self):
        return {str(uri(self)): to_json(ref.form_of(self))}

    def __ref__(self):
        raise RuntimeError("cannot reference a reflected Op; use Get, Put, Post, or Delete instead")

    def dtype(self):
        return self.__class__.__name__


class Get(Op):
    __uri__ = uri(op.Get)

    def __init__(self, form):
        self.rtype = _get_rtype(form, State)
        Op.__init__(self, form)

    def __args__(self):
        _, cxt = ref.form_of(self)
        return [cxt]

    def __form__(self):
        cxt, args = _maybe_first_arg(self)

        sig = inspect.signature(self.form)
        key_name = "key"
        if len(sig.parameters) > len(args):
            key_name = list(sig.parameters.keys())[len(args)]
            param = sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(form=URI(key_name)))

        cxt._return = self.form(*args)  # populate the Context

        validate(cxt, sig.parameters)

        return key_name, cxt

    def __ref__(self, name):
        rtype = self.rtype

        class GetRef(op.Get):
            def __call__(self, key):
                return rtype(form=ref.Get(self, key))

        return GetRef(URI(name))

    def __repr__(self):
        return f"GET Op with form {to_json(self)}"


class Put(Op):
    __uri__ = uri(op.Put)

    def __call__(self, key=None, value=None):
        return ref.Put(self, key, value)

    def __args__(self):
        _, _, cxt = ref.form_of(self)
        return [cxt]

    def __form__(self):
        cxt, args = _maybe_first_arg(self)

        sig = inspect.signature(self.form)
        key_name = "key"
        value_name = "value"

        if len(sig.parameters) == len(args):
            pass
        elif len(sig.parameters) - len(args) == 1:
            param_name = list(sig.parameters.keys())[-1]
            param = sig.parameters[param_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            if param_name in set(["key", "value"]):
                args.append(dtype(form=URI(param_name)))
            else:
                raise ValueError(f"{self.dtype()} argument {param_name} is ambiguous--use 'key' or 'value' instead")
        elif len(sig.parameters) - len(args) == 2:
            param_names = list(sig.parameters.keys())
            key_name = param_names[-2]
            param = sig.parameters[key_name]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(form=URI(key_name)))

            value_name = param_names[-1]
            param = sig.parameters[value_name]
            dtype = resolve_class(self.form, param.annotation, State)
            args.append(dtype(URI(value_name)))
        else:
            param_names = list(sig.parameters.keys())
            raise ValueError(
                f"{self.dtype()} accepts a maximum of three arguments: (cxt, key, value) (found {param_names})")

        cxt._return = self.form(*args)

        validate(cxt, sig.parameters)

        return key_name, value_name, cxt

    def __ref__(self, name):
        class PutRef(op.Put):
            def __call__(self, key=None, value=None):
                return Nil(ref.Put(key, value))

        return PutRef(URI(name))

    def __repr__(self):
        return f"PUT Op with form {self.form}"


class Post(Op):
    __uri__ = uri(op.Post)

    def __init__(self, form):
        self.rtype = _get_rtype(form, State)
        Op.__init__(self, form)

    def __args__(self):
        return [ref.form_of(self)]

    def __form__(self):
        cxt, args = _maybe_first_arg(self)

        sig = inspect.signature(self.form)
        kwargs = {}
        for name, param in list(sig.parameters.items())[len(args):]:
            dtype = resolve_class(self.form, param.annotation, State)
            kwargs[name] = dtype(form=URI(name))

        cxt._return = self.form(*args, **kwargs)

        validate(cxt, sig.parameters)

        return cxt

    def __ref__(self, name):
        sig = list(inspect.signature(self.form).parameters.items())
        rtype = self.rtype

        if sig:
            if sig[0][0] in ["cxt", "txn"]:
                sig = sig[1:]

        class PostRef(op.Post):
            def __call__(self, *args, **kwargs):
                params = parse_args(sig, *args, **kwargs)
                return rtype(form=ref.Post(self, params))

        return PostRef(URI(name))

    def __repr__(self):
        return f"POST Op with form {self.form}"


class Delete(Op):
    __uri__ = uri(op.Delete)

    def __args__(self):
        _, cxt = ref.form_of(self)
        return [cxt]

    def __form__(self):
        return Get.__form__(self)

    def __ref__(self, name):
        class DeleteRef(op.Delete):
            def __call__(self, key=None):
                return Nil(ref.Delete(self, key))

        return DeleteRef(URI(name))

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

    validate(cxt, sig.parameters)

    return cxt, args


def validate(cxt, provided):
    defined = set(provided)
    for name in provided:
        if name in cxt:
            raise RuntimeError(f"namespace collision: {name} in {cxt}")

    for name in cxt.form:
        def validate_ref(ref):
            if not hasattr(ref, "__uri__") and not isinstance(ref, URI):
                return

            ref = uri(ref)
            if ref.id() is not None and ref.id() not in defined:
                logging.info(f"{cxt} depends on undefined state {ref.id()}--is it part of a Closure?")

        form = cxt.form[name]
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
