import inspect

from ..uri import URI

from .functions import resolve_class


class MethodStub(object):
    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"cannot call {self} from a static context")

    def expand(self, header, name):
        raise NotImplementedError(f"{self.__class__.__name__}.expand")


class StateFunctionStub(MethodStub):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def expand(self, header, name):
        return self.dtype.expand(header, self.form, name)


class ReflectionStub(MethodStub):
    def __init__(self, form):
        self.form = form

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"cannot call the reflected method {self.form} from a static context")

    def expand(self, header, name):
        from ..state import State

        sig = tuple(inspect.signature(self.form).parameters.items())
        params = [header]

        if not sig or sig[0][0] != "self":
            raise RuntimeError(f"the first argument to an instance method must be 'self', not '{sig[0][0]}'")

        if sig[1][0] in ["cxt", "txn"]:
            raise RuntimeError(f"a reflected method must supply its own Context")

        for param_name, param in sig[1:]:
            dtype = resolve_class(self.form, param.annotation, State)
            params.append(dtype(form=URI(param_name)))

        yield name, self.form(*params)
