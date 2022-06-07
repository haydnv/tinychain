import inspect

from ..reflect import method, op, resolve_class
from ..scalar import op, ref
from ..state import State
from ..uri import URI

from .interface import Numeric


class Function(op.Post):
    def __init__(self, form):
        sig = inspect.signature(form)

        if not sig.parameters:
            raise ValueError(f"{form} is a constant, not a differentiable function")

        params = list(sig.parameters.items())

        i = 1 if params[0][0] in ["cxt", "txn"] else 0
        for name, param in params[i:]:
            dtype = resolve_class(self.form, param.annotation, State)
            if not inspect.isclass(dtype) or not issubclass(dtype, Numeric):
                raise TypeError(f"a differentiable function requires only numeric inputs, not {name}: {dtype}")

        op.Post.__init__(self, form)

    # TODO: deduplicate with Post.__ref__
    def __ref__(self, name):
        sig = list(inspect.signature(self.form).parameters.items())
        rtype = self.rtype

        if sig:
            if sig[0][0] in ["cxt", "txn"]:
                sig = sig[1:]

        class _FunctionRef(FunctionRef):
            def __call__(self, *args, **kwargs):
                params = op.parse_args(sig, *args, **kwargs)
                # TODO: chain rule
                return rtype(form=ref.Post(self, params))

        return _FunctionRef(URI(name))

    def __repr__(self):
        return f"POST Op with form {self.form}"


class FunctionRef(op.Post):
    pass


class StateFunction(method.Post):
    def __init__(self, header, form, name):
        params = list(inspect.signature(form).parameters.items())

        if not params or params[0][0] != "self":
            raise ValueError(f"{form} is missing a self parameter")

        i = 2 if params[1][0] in ["cxt", "txn"] else 1
        if i == len(params):
            raise ValueError(f"{form} is a constant, not a differentiable method")

        for name, param in params[i:]:
            dtype = resolve_class(form, param.annotation, State)
            if not inspect.isclass(dtype) or not issubclass(dtype, Numeric):
                raise TypeError(f"a differentiable method requires only numeric inputs, not {name}: {dtype}")

        method.Post.__init__(self, header, form, name)

    def __call__(self, *args, **kwargs):
        sig = list(inspect.signature(self.form).parameters.items())
        rtype = self.rtype

        if sig[1][0] in ["cxt", "txn"]:
            sig = sig[2:]
        else:
            sig = sig[1:]

        params = method.parse_args(sig, *args, **kwargs)
        # TODO: chain rule
        return rtype(form=ref.Post(self.subject(), params))
