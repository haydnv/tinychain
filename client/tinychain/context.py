"""Namespace utilities for :class:`Op` and :class:`Model` execution contexts."""

import inspect
import operator

from .app import Model
from .chain import Chain
from .generic import autobox
from .json import to_json
from .scalar.ref import args, form_of, get_ref, hex_id, is_literal, same_as, Ref
from .state import hash_of
from .state import State
from .uri import validate, URI


class _HashTable(object):
    SIZE = 100

    def __init__(self, hash_fn=hash, equivalence_fn=operator.eq):
        self._hash_fn = hash_fn
        self._equivalence_fn = equivalence_fn
        self._buckets = [[] for _ in range(self.SIZE)]

    def __contains__(self, key):
        bucket = self._buckets[self._hash_fn(key) % self.SIZE]
        for k, _ in bucket:
            if self._equivalence_fn(k, key):
                return True

    def __delitem__(self, key):
        i = self._hash_fn(key) % self.SIZE
        self._buckets[i] = [(k, v) for (k, v) in self._buckets[i] if not self._equivalence_fn(k, key)]

    def __getitem__(self, key):
        bucket = self._buckets[self._hash_fn(key) % self.SIZE]
        for k, v in bucket:
            if self._equivalence_fn(k, key):
                return v

        raise KeyError(f"{self} has no state named {key}")

    def __iter__(self):
        for bucket in self._buckets:
            for k, _v in bucket:
                yield k

    def __len__(self):
        return sum(len(bucket) for bucket in self._buckets)

    def __repr__(self):
        return f"HashTable{tuple(self)}"

    def __setitem__(self, key, value):
        assert self._equivalence_fn(key, key)
        assert self._equivalence_fn(value, value)

        bucket = self._buckets[self._hash_fn(key) % self.SIZE]
        if any(self._equivalence_fn(k, key) for k, _ in bucket):
            raise KeyError(f"{key} is already present in {self}")

        bucket.append((key, value))


class _Bijection(object):
    def __init__(self, hash_fn, equivalence_fn):
        self._by_key = _HashTable()
        self._by_value = _HashTable(hash_fn, equivalence_fn)

    def __contains__(self, key):
        return key in self._by_key

    def __delitem__(self, key):
        assert isinstance(key, str)
        if key in self._by_key:
            value = self._by_key[key]
            del self._by_key[key]
            del self._by_value[value]

        assert len(self._by_key) == len(self._by_value)

    def __getitem__(self, key):
        assert isinstance(key, str)
        if key in self._by_key:
            return self.value(key)
        else:
            raise KeyError(f"key {key} is not present in {self}")

    def __len__(self):
        return len(self._by_key)

    def __repr__(self):
        return f"Bijection({self._by_key})"

    def __setitem__(self, key, value):
        assert key

        if key in self._by_key:
            raise KeyError(f"key {key} is already present in {self}")
        elif value in self._by_value:
            raise KeyError(f"value {value} is already present in {self}")

        self._by_key[key] = value
        self._by_value[value] = key

        assert len(self._by_key) == len(self._by_value)

    def key(self, value):
        if value in self._by_value:
            return self._by_value[value]

    def value(self, key):
        if key in self._by_key:
            return self._by_key[key]


class Context(object):
    """A transaction context."""

    def __init__(self, copy_from=None):
        def hash_fn(state):
            if is_literal(state):
                return hex_id(state)
            else:
                return hash_of(state)

        object.__setattr__(self, "_ns", _Bijection(hash_fn, same_as))
        object.__setattr__(self, "_names", [])
        object.__setattr__(self, "_deps", [])

        if copy_from:
            assert isinstance(copy_from, Context)

            # TODO: only copy states from the old Context which the new Context depends on
            for name in copy_from._names[:-1]:
                self._names.append(name)
                self._ns[name] = copy_from._ns[name]

    def __bool__(self):
        return bool(self._names)

    def __contains__(self, name):
        return validate(name) in self._ns

    def __getattr__(self, name):
        name = validate(name)

        if name in self._ns:
            value = self._ns[name]
        else:
            raise AttributeError(f"Context has no such value: {name}")

        if hasattr(value, "__ref__"):
            return get_ref(value, name)
        else:
            return URI(name)

    def __getitem__(self, selector):
        keys = list(self)[selector]
        if isinstance(keys, list):
            return [getattr(self, key) for key in keys]
        else:
            return getattr(self, keys)

    def __iter__(self):
        return iter(self._names)

    def __json__(self):
        def reference(state, top_level=False):
            if hasattr(state, "__uri__") and state.__uri__.is_id():
                return state

            # if this state already has a name assigned, just return that name
            if isinstance(state, dict):
                return {key: reference(state[key]) for key in state}
            elif isinstance(state, list):
                return [reference(item) for item in state]
            elif isinstance(state, tuple):
                return tuple(reference(item) for item in state)
            elif self._ns.key(state):
                # unless its form is explicitly requested
                if not top_level:
                    name = self._ns.key(state)
                    return getattr(self, name)

            if isinstance(state, Ref):
                deps = [reference(arg) for arg in args(state)]
                return type(state)(*deps)
            elif isinstance(state, URI):
                # TODO: it shouldn't be necessary to reference private instance variables of URI here
                subject = reference(state._subject)
                return URI(subject, *state._path)
            elif isinstance(state, (Chain, Model)):
                return state
            elif isinstance(state, State):
                form = reference(form_of(state), top_level)
                return type(state)(form=form)
            else:
                return state

        form = []

        for name in self._deps + self._names:
            state = self._ns[name]
            state_ref = reference(state, True)
            form.append((name, state_ref))

        return to_json(form)

    def __len__(self):
        return len(self._ns)

    def __setattr__(self, name, state):
        if state is self:
            raise ValueError(f"cannot assign transaction Context to itself")

        name = validate(name, state)

        if name in self:
            raise ValueError(f"Context already has a value named {name} (contents are {self._ns}")
        elif self._ns.key(state):
            raise ValueError(f"cannot rename {state} from {self._ns.key(state)} to {name}")

        state = autobox(state)
        self.deanonymize(state, name)
        self._ns[name] = state
        self._names.append(name)

    def __repr__(self):
        data = list(self)
        return f"Op context with data {data}"

    def assign(self, state, name_hint):
        """
        Auto-assign a name based on `name_hint` to the given `state`.

        If `state` already has a name in this :class:`Context`, this is a no-op.
        """

        name_hint = validate(name_hint)

        if self._ns.key(state):
            return

        state = autobox(state)
        name_hint = str(name_hint[1:]) if name_hint.startswith('$') else str(name_hint)

        if isinstance(state, dict):
            for key in state:
                self.assign(state[key], f"{name_hint}.{key}")
            return
        elif isinstance(state, (list, tuple)):
            for i in range(len(state)):
                self.assign(state[i], f"{name_hint}.{i}")
            return

        if name_hint not in self:
            self._ns[name_hint] = state
            self._deps.append(name_hint)
            return

        i = 1
        while f"{name_hint}_{i}" in self:
            i += 1

        name_hint = f"{name_hint}_{i}"
        self._ns[name_hint] = state
        self._deps.append(name_hint)

    def deanonymize(self, state, name_hint):
        """Assign auto-generated names based on the given `name_hint` to the dependencies of the given `state`."""

        name_hint = validate(name_hint)

        if isinstance(state, Context):
            raise ValueError(f"cannot deanonymize an Op context itself")
        elif inspect.isclass(state):
            return

        if hasattr(state, "__ns__"):
            state.__ns__(self, name_hint)
        elif isinstance(state, dict):
            for key in state:
                if not isinstance(key, str):
                    raise KeyError(f"invalid key for autoboxed Map: {key}")

                self.deanonymize(state[key], name_hint + f".{key}")
        elif isinstance(state, (list, tuple)):
            for i, item in enumerate(state):
                self.deanonymize(item, name_hint + f".{i}")

    def items(self):
        """Iterate over the named states in this :class:`Context`."""

        yield from ((name, getattr(self, name)) for name in self)
