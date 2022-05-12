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
        from .scalar.ref import get_ref
        from .uri import URI

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
        from .uri import uri

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


def print_json(state_or_ref):
    """Pretty-print the JSON representation of the given `state_or_ref` to stdout."""

    print(json.dumps(to_json(state_or_ref), indent=4))


def to_json(obj):
    """Return a JSON-encodable representation of the given state or reference."""

    from .uri import uri

    if inspect.isgenerator(obj):
        raise ValueError(f"the Python generator {obj} is not JSON serializable")

    if callable(obj) and not hasattr(obj, "__json__"):
        raise ValueError(f"Python callable {obj} is not JSON serializable; consider using a decorator like @get")

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

    if isinstance(state, Context):
        raise ValueError(f"cannot deanonymize an Op context itself")

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
