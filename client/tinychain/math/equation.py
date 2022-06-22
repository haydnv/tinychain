import inspect

from ..context import deanonymize, Context
from ..reflect import method, op, get_rtype, resolve_class
from ..scalar import ref
from ..state import State, StateRef
from ..uri import URI

from .interface import Numeric
from .operator import derivative_of, gradients, Gradients, Operator


class FunctionCall(Operator):
    def __ns__(self, context, name_hint):
        deanonymize(self.subject, context, name_hint + "_subject")
        deanonymize(self.args, context, name_hint + "_args")

        if not ref.is_ref(self.subject):
            self.subject = context.assign(self.subject, name_hint + "_subject")
            assert isinstance(self.subject, StateRef)

        for name in self.args:
            if ref.is_op_ref(self.args[name]):
                self.args[name] = context.assign(self.args[name], f"{name_hint}_{name}")

    def __repr__(self):
        return f"call {self.subject} with inputs {self.args}"

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self):
        return ref.Post(self.subject, self.args)

    def backward(self, variable=None):
        d = derivative_of(self.subject)

        if isinstance(d, (Function, StateFunction)):
            return derivative_of(self.subject)(**self.args)
        else:
            return d

    def gradients(self, loss):
        grads = Gradients()

        for arg in self.args.values():
            grads.update(gradients(arg, loss * self.backward(arg)))

        subject = self.subject
        while isinstance(subject, StateRef):
            subject = subject.state

        assert isinstance(subject, (Function, StateFunction))

        if isinstance(subject, StateFunction):
            for member_name in dir(subject.header):
                if member_name.startswith('_'):
                    continue

                member = getattr(subject.header, member_name)
                if isinstance(member, Numeric):
                    grads.update(gradients(member, loss * self.backward(member)))

        return grads


class Function(op.Post):
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

        return FunctionRef(self, name)

    def __repr__(self):
        return f"differentiable POST Op with form {self.graph}"

    def derivative(self):
        graph = Context()
        graph._return = derivative_of(self.graph[-1])
        return Function(self.sig, graph, type(graph[-1]))


# TODO: dedupe with op.Post
class NativeFunction(Function):
    def __init__(self, form):
        self.sig = list(inspect.signature(form).parameters.items())
        self.form = form
        self.rtype = get_rtype(form, State)

        if not self.sig:
            raise ValueError(f"{self.form} is a constant, not a differentiable function")

        if not inspect.isclass(self.rtype) or not issubclass(self.rtype, Numeric):
            raise TypeError(f"a differentiable function {form} must return a numeric type, not {self.rtype}")

    def __form__(self):
        first_param_name = self.sig[0][0]

        if first_param_name == "self":
            raise TypeError(f"{self.form} is an instance method, not a stateless Op")
        elif first_param_name == "cls":
            raise TypeError(f"{self.form} is a classmethod, not a stateless Op")

        placeholders = []
        i = 1 if first_param_name in ["cxt", "txn"] else 0
        for name, param in self.sig[i:]:
            if param.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise RuntimeError(f"a user-defined Op supports positional or keyword arguments, not {param}")

            dtype = resolve_class(self.form, param.annotation, State)
            if inspect.isclass(dtype) and issubclass(dtype, Numeric):
                placeholders.append(dtype(form=name if isinstance(name, URI) else URI(name)))
            else:
                raise TypeError(f"a differentiable function requires only numeric inputs, not {name}: {dtype}")

        context = Context()
        args = [context] + placeholders if first_param_name in ["cxt", "txn"] else placeholders
        context._return = self.form(*args)

        return context

    def __repr__(self):
        return f"differentiable POST Op with form {self.form}"

    def derivative(self):
        form = ref.form_of(self)

        graph = Context()
        graph._return = derivative_of(form[-1])

        return Function(self.sig, graph, type(graph[-1]))


class StateFunction(method.Post):
    @classmethod
    def expand(cls, header, form, name):
        function = NativeStateFunction(header, form, name)
        yield name, function

        degree = 1
        form = ref.form_of(function)

        graph = Context()
        graph._return = derivative_of(form[-1])

        derivative = StateFunction(header, degree, name, function.sig, graph, type(graph[-1]))

        yield f"d_{name}", derivative

    def __init__(self, header, degree, name, sig, graph, rtype):
        self.header = header
        self.degree = degree
        self.name = name
        self.sig = sig
        self.graph = graph
        self.rtype = rtype

    def __call__(self, *args, **kwargs):
        rtype = self.rtype

        if self.sig[1][0] in ["cxt", "txn"]:
            sig = self.sig[2:]
        else:
            sig = self.sig[1:]

        params = method.parse_args(sig, *args, **kwargs)
        return rtype(form=FunctionCall(self, params))

    def __form__(self):
        return self.graph

    # TODO: deduplicate with Post.__ref__
    def __ref__(self, name):
        if self.sig[1][0] in ["cxt", "txn"]:
            sig = self.sig[2:]
        else:
            sig = self.sig[1:]

        class StateFunctionRef(StateRef):
            def __call__(self, *args, **kwargs):
                params = op.parse_args(sig, *args, **kwargs)
                return self.state.rtype(form=FunctionCall(self, params))

            def derivative(self):
                return self.state.derivative()

        return StateFunctionRef(self, name)

    def derivative(self):
        if self.degree == 0:
            name = f"d_{self.name}"
        else:
            name = f"d{self.degree + 1}_{self.name}"

        return getattr(self.header, name)


# TODO: dedupe with method.Post
class NativeStateFunction(StateFunction):
    def __init__(self, header, form, name):
        self.degree = 0
        self.header = header
        self.name = name
        self.sig = list(inspect.signature(form).parameters.items())
        self.form = form
        self.rtype = get_rtype(form, State)

        if not self.sig or self.sig[0][0] != "self":
            raise ValueError(f"{form} is missing a self parameter")

        i = 2 if self.sig[1][0] in ["cxt", "txn"] else 1
        if i == len(self.sig):
            raise ValueError(f"{form} is a constant, not a differentiable method")

        for name, param in self.sig[i:]:
            dtype = resolve_class(form, param.annotation, State)
            if not inspect.isclass(dtype) or not issubclass(dtype, Numeric):
                raise TypeError(f"a differentiable method requires only numeric inputs, not {name}: {dtype}")

        if not inspect.isclass(self.rtype) or not issubclass(self.rtype, Numeric):
            raise TypeError(f"a differentiable method must return a numeric type, not {self.rtype}")

    def __form__(self):
        cxt, args = method.first_params(self)

        kwargs = {}
        for name, param in self.sig[len(args):]:
            dtype = State
            if param.default is inspect.Parameter.empty:
                if param.annotation:
                    dtype = param.annotation
            elif isinstance(param.default, State):
                dtype = type(param.default)

            dtype = resolve_class(self.form, dtype, State)
            kwargs[name] = dtype(form=name if isinstance(name, URI) else URI(name))

        cxt._return = self.form(*args, **kwargs)

        for name in kwargs.keys():
            if name in cxt:
                raise RuntimeError(f"namespace collision: {name} in {self.form}")

        return cxt
