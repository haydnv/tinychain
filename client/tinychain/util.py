"""Utility data structures and functions."""

import inspect

from collections import OrderedDict


class Context(object):
    """A transaction context."""

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

    def __deps__(self):
        provided = set(URI(name) for name in self.form.keys())

        deps = set()
        for state in self.form.values():
            deps.update(requires(state))

        return deps - provided

    def __getattr__(self, name):
        if name in self.form:
            value = self.form[name]
            if isinstance(value, type):
                from .state import Class
                return Class(URI(name))
            elif hasattr(value, "__ref__"):
                return get_ref(value, name)
            else:
                return URI(name)
        else:
            raise ValueError(f"Context has no such value: {name}")

    def __json__(self):
        return to_json(list(self.form.items()))

    def __setattr__(self, name, state):
        if state is self:
            raise ValueError(f"cannot assign transaction Context to itself")

        deanonymize(state, self)

        if name in self.form:
            raise ValueError(f"Context already has a value named {name} (contents are {self.form}")
        else:
            self.form[name] = state


def form_of(state):
    """Return the form of the given state."""

    if hasattr(state, "__form__"):
        if callable(state.__form__) and not inspect.isclass(state.__form__):
            return state.__form__()
        else:
            return state.__form__
    else:
        raise ValueError(f"{state} has no form")


def get_ref(subject, name):
    """Return a named reference to the given state."""

    if hasattr(subject, "__ref__"):
        return subject.__ref__(name)
    else:
        return subject


class URI(object):
    """
    An absolute or relative link to a Tinychain state.
    
    Examples:
        .. highlight:: python
        .. code-block:: python

            URI("https://example.com/myservice/value_name")
            URI("$other_state/method_name")
            URI("/state/scalar/value/none")
    """

    def __init__(self, root, path=[]):
        assert root is not None

        if root.startswith("$$"):
            raise ValueError(f"invalid reference: {root}")

        self._root = str(root)
        self._path = path

    def __add__(self, other):
        if other == "/":
            return self
        else:
            return URI(str(self) + other)

    def __radd__(self, other):
        return URI(other) + str(self)

    def __deps__(self):
        if "://" in self._root or self.startswith('/'):
            return set()
        elif '/' in self._root:
            return set([URI(self._root[:self._root.index('/')])])
        else:
            return set([URI(self._root)])

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __json__(self):
        return {str(self): []}

    def __repr__(self):
        return str(self)

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

    def append(self, name):
        """
        Append a segment to this `URI`.
        
        Example:
            .. highlight:: python
            .. code-block:: python

                value = OpRef.Get(URI("http://example.com/myapp").append("value_name"))
        """

        if not str(name):
            return self

        return URI(str(self), [name])

    def is_id(self):
        """Return `True` if this URI is a simple ID, like `$foo`."""

        return '/' not in str(self)

    def host(self):
        """Return the host segment of this `URI`, if present."""

        if "://" not in self._root:
            return None

        start = self._root.index("://") + 3
        if '/' not in self._root[start:]:
            return self._root[start:]

        end = (
            self._root.index(':', start) if ':' in self._root[start:]
            else self._root.index('/', start))

        if end > start:
            return self._root[start:end]
        else:
            return self._root[start:]

    def path(self):
        """Return the path segment of this `URI`."""

        if "://" not in str(self._root):
            return URI(self._root) + "/".join(self._path)

        start = self._root.index("://")

        if '/' not in self._root[(start + 3):]:
            return None

        start = self._root.index('/', start + 3)

        prefix = URI(self._root[start:])
        if self._path:
            return prefix + "/".join(self._path)
        else:
            return prefix

    def port(self):
        """Return the port of this `URI`, if present."""

        prefix = self.protocol() + "://" if self.protocol() else ""
        prefix += self.host() if self.host() else ""
        if prefix and self._root[len(prefix)] == ':':
            end = self._root.index('/', len(prefix))
            return int(self._root[len(prefix) + 1:end])

    def protocol(self):
        """Return the protocol of this `URI` (e.g. "http"), if present."""

        if "://" in self._root:
            i = self._root.index("://")
            if i > 0:
                return self._root[:i]

    def subject(self):
        """Return only the ID portion of this `URI`, or `None` in the case of a link."""

        if "://" in self._root:
            return None
        elif self._root.startswith('/'):
            return None

        uri = str(self)
        if '/' in uri:
            return URI(uri[:uri.index('/')])
        else:
            return URI(uri)

    def startswith(self, prefix):
        return str(self).startswith(str(prefix))


def requires(subject):
    """Return a set of the IDs of the dependencies required to resolve the given state."""

    if hasattr(subject, "__deps__"):
        return subject.__deps__()

    deps = set()

    if isinstance(subject, list) or isinstance(subject, tuple):
        for item in subject:
            deps.update(requires(item))

    elif isinstance(subject, dict):
        for item in subject.values():
            deps.update(requires(item))

    return deps


def uri(subject):
    """Return the `URI` of the given state."""

    if hasattr(subject, "__uri__"):
        return subject.__uri__
    elif isinstance(subject, URI):
        return subject
    elif hasattr(type(subject), "__uri__"):
        return uri(type(subject))
    else:
        raise AttributeError(f"{subject} has no URI")


def use(cls):
    """Return an instance of the given class with callable methods."""

    if hasattr(cls, "__use__"):
        return cls.__use__()
    else:
        cls()


def to_json(obj):
    """Return a JSON-encodable representation of the given state or reference."""

    if inspect.ismethod(obj):
        if not hasattr(obj, "__json__"):
            raise ValueError(
                f"Python method {obj} is not JSON serializable; "
                + "try using a decorator like @get_method")

    if inspect.isclass(obj):
        if hasattr(type(obj), "__json__"):
            return type(obj).__json__(obj)
        elif hasattr(obj, "__uri__"):
            return to_json({str(uri(obj)): {}})

    if hasattr(obj, "__json__"):
        return obj.__json__()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(i) for i in obj]
    elif isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    else:
        return obj


def deanonymize(state, context):
    """Assign an auto-generated name to the given state within the given context."""

    if inspect.isclass(state):
        return
    elif hasattr(state, "__ns__"):
        state.__ns__(context)
    elif isinstance(state, tuple) or isinstance(state, list):
        for item in state:
            deanonymize(item, context)
    elif isinstance(state, dict):
        for key in state:
            deanonymize(state[key], context)
