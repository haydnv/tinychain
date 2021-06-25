import inspect

from tinychain import op, ref
from tinychain.state import State
from tinychain.util import form_of, to_json, uri, Context, URI
from tinychain.value import Nil, Value

from . import resolve_class


EMPTY = inspect.Parameter.empty


class Method(object):
    __uri__ = uri(op.Op)

    def __init__(self, header, form, name):
        self.header = header
        self.form = form
        self.name = name

    def __json__(self):
        return {str(uri(self)): to_json(form_of(self))}

    def dtype(self):
        return self.__class__.__name__


class Get(Method):
    __uri__ = uri(op.Get)

    def __call__(self, key=None):
        rtype = inspect.signature(self.form).return_annotation
        rtype = resolve_class(self.form, rtype, Nil)
        return rtype(ref.Get(uri(self.header).append(self.name), key))

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) < 1 or len(parameters) > 3:
            raise ValueError(f"{self.dtype()} takes 1-3 arguments: (self, cxt, key)")

        args = [self.header]

        cxt = Context()
        if len(parameters) > 1:
            _check_context_param(parameters[1])
            args.append(cxt)

        key_name = "key"
        if len(parameters) == 3:
            key_name, param = parameters[2]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

        cxt._return = self.form(*args)  # populate the Context
        return key_name, cxt


class Put(Method):
    __uri__ = uri(op.Put)

    def __call__(self, key, value):
        return ref.Put(uri(self.header).append(self.name), key, value)

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())
        if len(parameters) > 4:
            raise ValueError("a PUT method has a maximum of four parameters")

        args = [self.header]

        cxt = Context()
        if len(parameters) > 1:
            _check_context_param(parameters[1])
            args.append(cxt)

        key_name = "key"
        value_name = "value"

        if len(parameters) == 3:
            name, param = parameters[2]
            if name == key_name:
                dtype = resolve_class(self.form, param.annotation, Value)
                args.append(dtype(URI(key_name)))
            elif name == value_name:
                dtype = resolve_class(self.form, param.annotation, State)
                args.append(dtype(URI(value_name)))
            else:
                raise ValueError(
                    f"a PUT method with three parameters must specify either 'key' or 'value', not '{name}'")

        if len(parameters) == 4:
            key_name, param = parameters[2]
            dtype = resolve_class(self.form, param.annotation, Value)
            args.append(dtype(URI(key_name)))

            value_name, param = parameters[3]
            dtype = resolve_class(self.form, param.annotation, State)
            args.append(dtype(URI(value_name)))

        cxt._return = self.form(*args)
        return key_name, value_name, cxt


class Post(Method):
    __uri__ = uri(op.Post)

    def __call__(self, **params):
        rtype = inspect.signature(self.form).return_annotation
        rtype = resolve_class(self.form, rtype, Nil)
        return rtype(ref.Post(uri(self.header).append(self.name), **params))

    def __form__(self):
        sig = inspect.signature(self.form)
        parameters = list(sig.parameters.items())

        if len(parameters) == 0:
            raise ValueError(f"{self.dtype()} has at least one argument: (self, cxt, name1=val1, ...)")

        args = [self.header]

        cxt = Context()
        if len(parameters) > 1:
            _check_context_param(parameters[1])
            args.append(cxt)

        kwargs = {}
        for name, param in parameters[2:]:
            dtype = resolve_class(self.form, param.annotation, State)
            kwargs[name] = dtype(URI(name))

        cxt._return = self.form(*args, **kwargs)
        return cxt


class Delete(Method):
    __uri__ = uri(op.Delete)

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
