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
        if name in self.form:
            value = self.form[name]
            if hasattr(value, "__ref__"):
                return type(value)(URI(name))
            else:
                from .state import IdRef
                return IdRef(name)
        else:
            raise ValueError(f"Context has no such value: {name}")

    def __json__(self):
        return to_json(list(self.form.items()))

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


def ref(subject):
    if hasattr(subject, "__ref__"):
        return subject.__ref__


class URI(object):
    def __init__(self, root, path=[]):
        assert root is not None

        self._root = root
        self._path = path

    def append(self, name):
        return URI(str(self), [name])

    def host(self):
        if "://" not in self._root:
            return None

        start = self._root.index("://") + 3
        end = self._root.index('/', start)
        if end > start:
            return self._root[start:end]
        else:
            return self._root[start:]

    def path(self):
        if "://" not in self._root:
            return self._root

        start = self._root.index("://")
        start = self._root.index('/', start + 3)
        return self._root[start:] + self._path

    def protocol(self):
        i = self._root.index("://")
        if i > 0:
            return self._root[:i]

    def __add__(self, other):
        return URI(str(self) + other)

    def __json__(self):
        return {str(self): []}

    def __str__(self):
        root = str(self._root)
        if root.startswith('/') or root.startswith('$') or "://" in root:
            pass
        else:
            root = f"${root}"

        if self._path:
            path = "/".join(self._path)
            return f"{root}/{path}"
        else:
            return root


def uri(subject):
    if hasattr(subject, "__uri__"):
        return subject.__uri__
    elif isinstance(subject, URI):
        return subject
    elif isinstance(ref(subject), URI):
        return ref(subject)
    elif hasattr(type(subject), "__ref__") or hasattr(type(subject), "__uri__"):
        return uri(type(subject))
    else:
        raise AttributeError(f"{subject} has no URI")


def use(cls):
    if hasattr(cls, "__use__"):
        return cls.__use__()
    else:
        cls


def to_json(obj):
    if inspect.isclass(obj):
        if hasattr(type(obj), "__json__"):
            return type(obj).__json__(obj)

    if hasattr(obj, "__json__"):
        return obj.__json__()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(i) for i in obj]
    elif isinstance(obj, dict):
        return {to_json(k): to_json(v) for k, v in obj.items()}
    else:
        return obj

