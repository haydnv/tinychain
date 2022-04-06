"""Utility data structures and functions."""

import inspect
import json
import logging

from collections import OrderedDict


class Context(object):
    """A transaction context."""

    def __init__(self, context=None):
        object.__setattr__(self, "_form", OrderedDict())

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

    def __contains__(self, item):
        return self._get_name(item) in self.form

    def __dbg__(self):
        return [self.form[next(reversed(self.form))]] if self.form else []

    def __getattr__(self, name):
        name = self._get_name(name)

        if name in self.form:
            value = self.form[name]
            if hasattr(value, "__ref__"):
                return get_ref(value, name)
            else:
                logging.info(f"context attribute {value} has no __ref__ method")
                return URI(name)
        else:
            raise AttributeError(f"Context has no such value: {name}")

    def __getitem__(self, selector):
        if isinstance(selector, slice):
            cxt = Context()
            keys = list(self.form.keys())
            for name in keys[selector]:
                setattr(cxt, name, self.form[name])

            return cxt
        else:
            return getattr(self, selector)

    def __json__(self):
        return to_json(list(self.form.items()))

    def __len__(self):
        return len(self.form)

    def __setattr__(self, name, state):
        from .state import State

        if state is self:
            raise ValueError(f"cannot assign transaction Context to itself")
        elif isinstance(state, dict):
            from .generic import Map
            state = Map(state)
        elif isinstance(state, (tuple, list)):
            from .generic import Tuple
            state = Tuple(state)
        elif isinstance(state, str):
            from .scalar.value import String
            state = String(state)
        elif not isinstance(state, State) and hasattr(state, "__iter__"):
            logging.warning(f"state {name} is set to {state}, which does not support URI assignment; " +
                            "consider a Map or Tuple instead")

        name = self._get_name(name)

        deanonymize(state, self)

        if name in self.form:
            raise ValueError(f"Context already has a value named {name} (contents are {self.form}")
        else:
            self.form[name] = state

    def __repr__(self):
        data = list(self.form.keys())
        return f"Op context with data {data}"

    def _get_name(self, item):
        if hasattr(item, "__uri__"):
            if uri(item).id() != uri(item):
                raise ValueError(f"invalid name: {item}")
            else:
                return uri(item).id()
        else:
            return str(item)

    @property
    def form(self):
        return self._form


def args(state_or_ref):
    """Return the references needed to resolve the given `state_or_ref`."""

    if form_of(state_or_ref) is not state_or_ref:
        return args(form_of(state_or_ref))

    return state_or_ref.__args__() if hasattr(state_or_ref, "__args__") else []


def depends_on(state_or_ref):
    """Return the set of all compile-time dependencies of the given `state_or_ref`"""

    if form_of(state_or_ref) is not state_or_ref:
        return depends_on(form_of(state_or_ref))

    from . import reflect
    from .scalar.ref import Ref

    if independent(state_or_ref):
        print("independent", state_or_ref)
        return [state_or_ref]
    else:
        print("not independent", state_or_ref)

    deps = []

    if isinstance(state_or_ref, Ref):
        for dep in args(state_or_ref):
            deps.extend(depends_on(dep))
    elif isinstance(state_or_ref, (list, tuple)):
        for dep in state_or_ref:
            deps.extend(depends_on(dep))
    elif isinstance(state_or_ref, dict):
        for dep in state_or_ref.values():
            deps.extend(depends_on(dep))

    return deps


def form_of(state, recurse=False):
    """
    Return the form of the given `state`.

    If `recurse` is `True`, this will proceed recursively through all intermediate type expectations.
    """

    if not hasattr(state, "__form__"):
        return state

    if callable(state.__form__) and not inspect.isclass(state.__form__):
        form = state.__form__()
    else:
        form = state.__form__

    if recurse:
        return form_of(form)
    else:
        return form


def get_ref(state, name):
    """Return a named reference to the given `state`."""

    name = URI(name)

    if inspect.isclass(state):
        def ctr(*args, **kwargs):
            return state(*args, **kwargs)

        ctr.__uri__ = name
        ctr.__json__ = lambda: to_json(uri(ctr))
        return ctr
    elif hasattr(state, "__ref__"):
        return state.__ref__(name)
    elif isinstance(state, dict):
        return {k: get_ref(v, name.append(k)) for k, v in state.items()}
    elif isinstance(state, (list, tuple)):
        return tuple(get_ref(item, name.append(i)) for i, item in enumerate(state))
    else:
        logging.debug(f"{state} has no __ref__ method")
        return state


def hex_id(state_or_ref):
    """
    Return a unique hexadecimal string identifying the given `state_or_ref` based on its memory address.

    This is similar to Python's built-in `id` function but has the advantage that it will still produce the correct
    unique ID even for wrapper types. For example:

    ```python
    x = 1
    n = Number(x)
    assert hex_id(n) == hex_id(x)  # this won't work with the built-in `id` function
    ```
    """

    if hasattr(state_or_ref, "__id__"):
        return state_or_ref.__id__()

    return format(id(state_or_ref), 'x')


def independent(state_or_ref):
    if form_of(state_or_ref) is not state_or_ref:
        return independent(form_of(state_or_ref))

    from . import reflect
    from .scalar.ref import Ref

    if isinstance(state_or_ref, (reflect.method.Method, reflect.op.Op)) or inspect.isclass(state_or_ref):
        return True

    if isinstance(state_or_ref, dict):
        return all(independent(value) for value in state_or_ref.values())
    elif isinstance(state_or_ref, (list, tuple)):
        return all(independent(item) for item in state_or_ref)
    elif isinstance(state_or_ref, Ref):
        return not args(state_or_ref)
    else:
        return True


class URI(object):
    """
    An absolute or relative link to a TinyChain state.
    
    Examples:
        .. highlight:: python
        .. code-block:: python

            URI("https://example.com/myservice/value_name")
            URI("$other_state/method_name")
            URI("/state/scalar/value/none")
    """

    def __init__(self, root, path=[]):
        root = str(root)
        if not root:
            raise ValueError(f"invalid URI root: {root}")

        if root.startswith("$$"):
            raise ValueError(f"invalid reference: {root}")
        elif root.startswith('$'):
            root = root[1:]

        self._root = root
        self._path = path

    def __add__(self, other):
        if other == "/":
            return self
        else:
            return URI(str(self) + other)

    def __radd__(self, other):
        return URI(other) + str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __gt__(self, other):
        return self.startswith(other) and len(self) > len(other)

    def __ge__(self, other):
        return self.startswith(other)

    def __len__(self):
        return len(str(self))

    def __lt__(self, other):
        return other > self

    def __le__(self, other):
        return other >= self

    def __hash__(self):
        return hash(str(self))

    def __json__(self):
        return {str(self): []}

    def __repr__(self):
        return str(self)

    def __str__(self):
        root = self._root
        if root.startswith('/') or "://" in root:
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
        Construct a new `URI` beginning with this `URI` and ending with the given `name` segment.

        Example:
            .. highlight:: python
            .. code-block:: python

                value = OpRef.Get(URI("http://example.com/myapp").append("value_name"))
        """

        if not str(name):
            return self

        return URI(str(self), [str(name)])

    def id(self):
        """Return the ID segment of this `URI`, if present."""

        this = str(self)
        if this.startswith('$'):
            if '/' in this:
                end = this.index("/")
                return this[1:end]
            else:
                return this[1:]

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

    def startswith(self, prefix):
        return str(self).startswith(str(prefix))


def print_json(obj):
    print(json.dumps(to_json(obj), indent=4))


def uri(subject):
    """Return the `URI` of the given state."""

    if isinstance(subject, URI):
        return subject

    if hasattr(subject, "__uri__"):
        if subject.__uri__ is None:
            raise ValueError(f"{subject} (a {type(subject)}) has a URI of {None}")

        return subject.__uri__

    if hasattr(type(subject), "__uri__"):
        return uri(type(subject))

    return None


def to_json(obj):
    """Return a JSON-encodable representation of the given state or reference."""

    err_native = "Python callable {} is not JSON serializable; consider using a decorator like @get"

    if callable(obj) and not hasattr(obj, "__json__"):
        raise ValueError(err_native.format(obj))

    if inspect.isclass(obj):
        if hasattr(type(obj), "__json__"):
            return type(obj).__json__(obj)
        elif hasattr(obj, "__uri__"):
            return to_json({str(uri(obj)): {}})

    if hasattr(obj, "__json__"):
        return obj.__json__()
    elif isinstance(obj, (list, tuple)):
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
    elif isinstance(state, (tuple, list)):
        for item in state:
            deanonymize(item, context)
    elif isinstance(state, dict):
        for key in state:
            deanonymize(state[key], context)
