"""Utility data structures and functions."""

import inspect
import json
import operator

from collections import OrderedDict


class _HashTable(object):
    SIZE = 100

    def __init__(self, hash_fn=hash, equivalence_fn=operator.eq):
        self._hash_fn = hash_fn
        self._equivalence_fn = equivalence_fn
        self._buckets = [[]] * self.SIZE

    def __contains__(self, key):
        bucket = self._buckets[self._hash_fn(key) % self.SIZE]
        for k, _ in bucket:
            if self._equivalence_fn(k, key):
                return True

    def __getitem__(self, key):
        bucket = self._buckets[self._hash_fn(key) % self.SIZE]
        for k, v in bucket:
            if self._equivalence_fn(k, key):
                return v

        raise KeyError(f"{key} is not present in {self}")

    def __setitem__(self, key, value):
        bucket = self._buckets[self._hash_fn(key) % self.SIZE]
        if any(self._equivalence_fn(k, key) for k, _ in bucket):
            raise KeyError(f"{key} is already present in {self}")

        bucket.append((key, value))


class _Bijection(object):
    def __init__(self, hash_fn, equivalence_fn):
        self._by_key = _HashTable()
        self._by_value = _HashTable(hash_fn, equivalence_fn)
        self._order = []

    def __contains__(self, key):
        return key in self._by_key

    def __getitem__(self, key):
        if key in self._by_key:
            return self.value(key)
        else:
            raise KeyError(f"key {key} is not present in {self}")

    def __iter__(self):
        return iter(self._order)

    def __setitem__(self, key, value):
        assert key

        if key in self._by_key:
            raise KeyError(f"key {key} is already present in {self}")
        elif value in self._by_value:
            raise KeyError(f"value {value} is already present in {self}")

        self._by_key[key] = value
        self._by_value[value] = key
        self._order.append(key)

    def items(self):
        for key in self._order:
            yield key, self[key]

    def key(self, value):
        if value in self._by_value:
            return self._by_value[value]

    def value(self, key):
        if key in self._by_key:
            return self._by_key[key]


class Context(object):
    """A transaction context."""

    def __init__(self):
        # TODO: handle multiple identical literal values with different semantic significance
        from .scalar.ref import same_as
        from .state import hash_of
        object.__setattr__(self, "_deps", _Bijection(hash_of, same_as))
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
        from .app import Model
        from .chain import Chain
        from .scalar.ref import args, form_of, Ref
        from .state import State
        from .uri import URI

        def reference(state, top_level=False):
            if hasattr(state, "__uri__") and state.__uri__.is_id():
                return state

            # if this state already has a name assigned, just return that name
            if not isinstance(state, (dict, list, tuple)) and self._deps.key(state):
                # unless its form is explicitly requested
                if not top_level:
                    return getattr(self, self._deps.key(state))

            if isinstance(state, Ref):
                deps = [reference(arg) for arg in args(state)]
                return type(state)(*deps)
            elif isinstance(state, URI):
                # TODO: it shouldn't be necessary to reference private instance variables of URI here
                return URI(reference(state._subject), *state._path)
            elif isinstance(state, (Chain, Model)):
                return state
            elif isinstance(state, State):
                form = reference(form_of(state), top_level)
                ref = type(state)(form=form)
                return ref
            elif isinstance(state, dict):
                ref = {key: reference(state[key]) for key in state}
                return ref
            elif isinstance(state, list):
                ref = [reference(item) for item in state]
                return ref
            elif isinstance(state, tuple):
                return tuple(reference(item) for item in state)
            else:
                return state

        form = []

        for name, state in self._deps.items():
            state_ref = reference(state, True)
            form.append((name, state_ref))

        for name, state in self._ns.items():
            state = reference(state, True)
            form.append((name, state))

        return to_json(form)

    def __len__(self):
        return len(self._ns)

    def __setattr__(self, name, state):
        if state is self:
            raise ValueError(f"cannot assign transaction Context to itself")

        name = str(name[1:]) if name.startswith('$') else str(name)

        if name in self:
            raise ValueError(f"Context already has a value named {name} (contents are {self._ns}")

        state = autobox(state)
        deanonymize(state, self, name)
        self._ns[name] = state

    def __repr__(self):
        data = list(self._ns.keys())
        return f"Op context with data {data}"

    def assign(self, state, name_hint):
        from .scalar.ref import same_as
        from .state import hash_of

        # TODO: handle multiple identical literal values with different semantic significance
        state_hash = hash_of(state)

        for name, present in self._ns.items():
            if state_hash == hash_of(present):
                if same_as(state, present):
                    return getattr(self, name)

        if self._deps.key(state):
            return getattr(self, self._deps.key(state))

        state = autobox(state)
        name_hint = str(name_hint[1:]) if name_hint.startswith('$') else str(name_hint)

        if isinstance(state, dict):
            for key in dict:
                self.assign(state[key], f"{name_hint}.{key}")
            return
        elif isinstance(state, (list, tuple)):
            for i in range(len(state)):
                self.assign(state[i], f"{name_hint}.{i}")
            return

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

    from .scalar.ref import Ref
    if isinstance(state, Ref):
        return state

    from .interface import Interface
    if isinstance(state, Interface):
        return state

    return state


def print_json(state_or_ref):
    """Pretty-print the JSON representation of the given `state_or_ref` to stdout."""

    print(json.dumps(to_json(state_or_ref), indent=4))


def to_json(state_or_ref):
    """Return a JSON-encodable representation of the given state or reference."""

    if inspect.isgenerator(state_or_ref):
        raise ValueError(f"the Python generator {state_or_ref} is not JSON serializable")

    if callable(state_or_ref) and not hasattr(state_or_ref, "__json__"):
        raise ValueError(f"Python callable {state_or_ref} is not JSON serializable; consider a decorator like @get")

    if inspect.isclass(state_or_ref):
        if hasattr(type(state_or_ref), "__json__"):
            return type(state_or_ref).__json__(state_or_ref)
        elif hasattr(state_or_ref, "__uri__"):
            return to_json({str(state_or_ref.__uri__): {}})

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

    if isinstance(state, Context):
        raise ValueError(f"cannot deanonymize an Op context itself")
    elif inspect.isclass(state):
        return

    if hasattr(state, "__ns__"):
        state.__ns__(context, name_hint)
    elif isinstance(state, dict):
        for key in state:
            deanonymize(state[key], context, name_hint + f".{key}")
    elif isinstance(state, (list, tuple)):
        for i, item in enumerate(state):
            deanonymize(item, context, name_hint + f".{i}")
