import inspect

from collections import OrderedDict


class Context(object):
    def __init__(self, context=None):
        object.__setattr__(self, "form", OrderedDict())

        if context:
            if isinstance(context, self.__class__):
                for name, value in context.form.items():
                    setattr(self, name, value)
            else:
                for name, value in dict(context).items():
                    setattr(self, name, value)

    def __add__(self, other):
        concat = Context(self)
        if isinstance(other, self.__class__):
            for name, value in other.form.items():
                setattr(concat, name, value)
        else:
            for name, value in dict(other).items():
                setattr(concat, name, value)

        return concat

    def __getattr__(self, name):
        if name in object.__getattribute__(self, "form"):
            return ref(self.form[name], name)
        else:
            raise ValueError(f"Context has no such value: {name}")

    def __json__(self):
        return [[to_json(name), to_json(spec)] for (name, spec) in self.form.items()]

    def __setattr__(self, name, value):
        if name in object.__getattribute__(self, "form"):
            raise ValueError(f"Context already has a value named {name}")
        else:
            self.form[name] = value


def form_of(op):
    if hasattr(op, "__form__"):
        return op.__form__()
    else:
        raise ValueError(f"{op} has no Context")


def ref(subject, name=None):
    if hasattr(subject, "__ref__"):
        return subject.__ref__(name)
    else:
        raise ValueError(f"{subject} does not support named references")


def to_json(obj):
    if inspect.isclass(obj) and hasattr(obj, "PATH"):
        return {obj.PATH: []}
    elif hasattr(obj, "__json__"):
        return obj.__json__()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(i) for i in obj]
    elif isinstance(obj, dict):
        return {to_json(k): to_json(v) for k, v in obj.items()}
    elif obj is None:
        return {"/state/value/none": [[]]}
    else:
        return obj

