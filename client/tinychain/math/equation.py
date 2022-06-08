import inspect

from ..context import Context
from ..reflect import method, op, get_rtype, resolve_class
from ..scalar import ref
from ..scalar.number import Bool, Float, Int
from ..state import State, StateRef
from ..uri import URI

from .interface import Numeric
from .operator import derivative_of, Operator


class FunctionCall(Operator):
    def __repr__(self):
        return f"call {self.subject} with inputs {self.args}"

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self):
        return ref.Post(self.subject, self.args)

    def backward(self, variable=None):
        d_function = derivative_of(self.subject)
        return d_function(**self.args)

    def gradients(self, loss):
        raise NotImplementedError


class Function(op.Post):
    @classmethod
    def reflect(cls, native):
        sig = list(inspect.signature(native).parameters.items())

        if not sig:
            raise ValueError(f"{native} is a constant, not a differentiable function")

        first_param_name = sig[0][0]

        if first_param_name == "self":
            raise TypeError(f"{native} is an instance method, not a stateless Op")
        elif first_param_name == "cls":
            raise TypeError(f"{native} is a classmethod, not a stateless Op")

        placeholders = []
        i = 1 if first_param_name in ["cxt", "txn"] else 0
        for name, param in sig[i:]:
            assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

            dtype = resolve_class(native, param.annotation, State)
            if inspect.isclass(dtype) and issubclass(dtype, Numeric):
                placeholders.append(dtype(form=URI(name)))
            else:
                raise TypeError(f"a differentiable function requires only numeric inputs, not {name}: {dtype}")

        context = Context()
        args = [context] + placeholders if first_param_name in ["cxt", "txn"] else placeholders
        context._return = native(*args)

        rtype = get_rtype(native, None)
        if not inspect.isclass(rtype) or not issubclass(rtype, Numeric):
            raise TypeError(f"a differentiable function {native} must return a numeric type, not {rtype}")

        return cls(sig, context, rtype)

    def __init__(self, sig, graph, rtype):
        self.sig = sig
        self.graph = graph
        self.rtype = rtype

    def __call__(self, *args, **kwargs):
        sig = self.sig[1:] if self.sig[0][0] in ["cxt", "txn"] else self.sig
        params = op.parse_args(sig, *args, **kwargs)
        return self.rtype(form=FunctionCall(self, params))

    def __form__(self):
        return self.graph

    # TODO: deduplicate with Post.__ref__
    def __ref__(self, name):
        sig = self.sig[1:] if self.sig[0][0] in ["cxt", "txn"] else self.sig

        class FunctionRef(StateRef):
            def __call__(self, *args, **kwargs):
                params = op.parse_args(sig, *args, **kwargs)
                return self.state.rtype(form=FunctionCall(self, params))

            def derivative(self):
                return self.state.derivative()

        return FunctionRef(self, URI(name))

    def __repr__(self):
        return f"POST Op with form {self.graph}"

    def derivative(self):
        graph = Context()
        graph._return = derivative_of(self.graph[-1])

        rtype = type(graph[-1])
        if rtype is bool:
            rtype = Bool
        elif rtype is float:
            rtype = Float
        elif rtype is int:
            rtype = Int

        return Function(self.sig, graph, rtype)


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
        rtype = self.rtype

        if self.sig[1][0] in ["cxt", "txn"]:
            sig = self.sig[2:]
        else:
            sig = self.sig[1:]

        params = method.parse_args(sig, *args, **kwargs)
        return rtype(form=FunctionCall(self.subject(), params))
