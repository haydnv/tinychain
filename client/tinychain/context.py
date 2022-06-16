"""Utility data structures and functions."""

import inspect
import json
import logging

from collections import OrderedDict


class Context(object):
    """A transaction context."""

    def __init__(self):
        object.__setattr__(self, "_deps", OrderedDict())
        object.__setattr__(self, "_ns", OrderedDict())

    def __contains__(self, name):
        name = str(name[1:]) if name.startswith('$') else str(name)
        return name in self._ns or name in self._deps

    def __getattr__(self, name):
        from .scalar.ref import get_ref
        from .uri import URI

        name = str(name[1:]) if name.startswith('$') else str(name)

        if name in self._ns:
            value = self._ns[name]
        elif name in self._deps:
            value = self._deps[name]
        else:
            raise AttributeError(f"Context has no such value: {name}")

        if hasattr(value, "__ref__"):
            return get_ref(value, name)
        else:
            logging.info(f"context attribute {value} has no __ref__ method")
            return URI(name)

    def __getitem__(self, selector):
        keys = list(self._ns.keys())[selector]
        if isinstance(keys, list):
            return [getattr(self, key) for key in keys]
        else:
            return getattr(self, keys)

    def __iter__(self):
        return iter(self._ns)

    def __json__(self):
        form = []
        form.extend(self._deps.items())
        form.extend(self._ns.items())

        return to_json(form)

    def __len__(self):
        return len(self._ns)

    def __setattr__(self, name, state):
        if state is self:
            raise ValueError(f"cannot assign transaction Context to itself")

        name = str(name[1:]) if name.startswith('$') else str(name)

        if name in self._ns:
            raise ValueError(f"Context already has a value named {name} (contents are {self._ns}")

        state = autobox(state)
        deanonymize(state, self, name)
        self._ns[name] = state

    def __repr__(self):
        data = list(self._ns.keys())
        return f"Op context with data {data}"

    def assign(self, state, name_hint):
        state = autobox(state)
        name_hint = str(name_hint[1:]) if name_hint.startswith('$') else str(name_hint)

        if name_hint not in self:
            self._deps[name_hint] = state
            return getattr(self, name_hint)

        i = 1
        while f"{name_hint}_{i}" in self:
            i += 1

        name_hint = f"{name_hint}_{i}"
        self._deps[name_hint] = state

        return getattr(self, name_hint)

    def items(self):
        yield from ((name, getattr(self, name)) for name in self._form)


def autobox(state):
    if isinstance(state, bool):
        from .scalar.number import Bool
        return Bool(state)
    elif isinstance(state, float):
        from .scalar.number import Float
        return Float(state)
    elif isinstance(state, int):
        from .scalar.number import Int
        return Int(state)
    elif isinstance(state, dict):
        from .generic import Map
        return Map(state)
    elif isinstance(state, (list, tuple)):
        from .generic import Tuple
        return Tuple(state)
    elif isinstance(state, str):
        from .scalar.value import String
        return String(state)

    from .state import State
    if isinstance(state, State):
        return state

    logging.debug(f"cannot autobox {state}")
    return state


def hashable(state):
    from .generic import Map, Tuple

    if isinstance(state, dict):
        return False
    elif isinstance(state, Map) and hasattr(state, "__iter__"):
        return False
    elif isinstance(state, (list, tuple)):
        return False
    elif isinstance(state, Tuple) and hasattr(state, "__iter__"):
        return False

    return True


def print_json(state_or_ref):
    """Pretty-print the JSON representation of the given `state_or_ref` to stdout."""

    print(json.dumps(to_json(state_or_ref), indent=4))


def to_json(state_or_ref):
    """Return a JSON-encodable representation of the given state or reference."""

    from .uri import URI

    if inspect.isgenerator(state_or_ref):
        raise ValueError(f"the Python generator {state_or_ref} is not JSON serializable")

    if callable(state_or_ref) and not hasattr(state_or_ref, "__json__"):
        raise ValueError(f"Python callable {state_or_ref} is not JSON serializable; consider a decorator like @get")

    if inspect.isclass(state_or_ref):
        if hasattr(type(state_or_ref), "__json__"):
            return type(state_or_ref).__json__(state_or_ref)
        elif hasattr(state_or_ref, "__uri__"):
            return to_json({str(URI(state_or_ref)): {}})

    if hasattr(state_or_ref, "__json__"):
        return state_or_ref.__json__()
    elif isinstance(state_or_ref, (list, tuple)):
        return [to_json(i) for i in state_or_ref]
    elif isinstance(state_or_ref, dict):
        return {str(k): to_json(v) for k, v in state_or_ref.items()}
    else:
        return state_or_ref


def deanonymize(state, context, name_hint):
    """Assign an auto-generated name based on the given `name_hint` to the given state within the given context."""

    from .scalar.ref import is_literal

    if isinstance(state, Context):
        raise ValueError(f"cannot deanonymize an Op context itself")
    elif is_literal(state):
        return

    if inspect.isclass(state):
        return
    elif hasattr(state, "__ns__"):
        state.__ns__(context, name_hint)
    elif isinstance(state, dict):
        for key in state:
            deanonymize(state[key], context, name_hint + f".{key}")
    elif isinstance(state, (list, tuple)):
        for i, item in enumerate(state):
            deanonymize(item, context, name_hint + f".{i}")
