import typing

from .interface import Functional
from .scalar.bound import Range
from .scalar.ref import deref, form_of, get_ref, is_literal, same_as, Get, Post
from .scalar.value import Id
from .state import State, StateRef
from .uri import uri
from .context import to_json


# TODO: implement `Functional` for `Map`
class Map(State):
    """A key-value map whose keys are `Id`s and whose values are `State` s."""

    __uri__ = uri(State) + "/map"
    __spec__ = typing.Dict[Id, State]

    @staticmethod
    def _parse_args(args, kwargs):
        form = None
        spec = None

        if args and "form" in kwargs:
            raise ValueError(f"Map got duplicate arguments for 'form': {args[0]}, {kwargs['form']}")

        if len(args) > 2:
            raise ValueError(f"Map.__init__ got unexpected arguments {args} {kwargs}")

        if "form" in kwargs:
            form = kwargs["form"]
            del kwargs["form"]
        elif len(args) >= 1:
            form = args[0]

        if len(args) > 1:
            spec = args[1]

        if not form:
            form = kwargs

        while isinstance(form, Map):
            form = form_of(form)

        if not spec:
            if hasattr(form, "__spec__"):
                spec = form.__spec__

            if form and isinstance(form, dict) and any(issubclass(type(v), State) for v in form.values()):
                spec = {k: type(v) if issubclass(type(v), State) else State for k, v in form.items()}

        return form, spec

    @classmethod
    def expect(cls, spec):
        class _Map(cls):
            __spec__ = spec

            def __contains__(self, key):
                return key in spec

            def __len__(self):
                return len(spec)

            def __iter__(self):
                return iter(spec)

        return _Map

    def __new__(*args, **kwargs):
        cls = args[0]
        form, spec = Map._parse_args(args[1:], kwargs)

        if isinstance(form, Map):
            return State.__new__(cls)
        elif spec:
            return State.__new__(cls.expect(spec))
        else:
            return State.__new__(cls)

    def __init__(self, *args, **kwargs):
        form, spec = Map._parse_args(args, kwargs)
        State.__init__(self, form)

    def __eq__(self, other):
        return self.eq(other)

    def __getitem__(self, key):
        if hasattr(form_of(self), "__getitem__"):
            if uri(self) == uri(self.__class__):
                return form_of(self)[key]
            else:
                return get_ref(form_of(self)[key], uri(self).append(key))
        elif isinstance(self.__spec__, dict):
            if key in self.__spec__:
                rtype = self.__spec__[key]
            else:
                raise KeyError(f"{self} does not contain any entry for {key}")
        else:
            rtype = typing.get_args(self.__spec__)[1]

        return self._get("", key, rtype)

    def __json__(self):
        return to_json(form_of(self))

    def __ne__(self, other):
        return self.ne(other)

    def __ref__(self, name):
        return self.__class__(form=MapRef(self, name))

    def eq(self, other):
        """Return a `Bool` indicating whether all the keys and values in this map are equal to the given `other`."""

        if same_as(self, other):
            return True

        from .scalar.number import Bool
        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """Return a `Bool` indicating whether the keys and values in this `Map` are not equal to the given `other`."""

        return self.eq(other).logical_not()

    def len(self):
        """Return the number of elements in this `Map`."""

        from .scalar.number import UInt
        return self._get("len", rtype=UInt)


class MapRef(StateRef):
    def __getitem__(self, key):
        return self.state[key]


class Tuple(State, Functional):
    """A tuple of `State` s."""

    __uri__ = uri(State) + "/tuple"
    __spec__ = (State, ...)

    @classmethod
    def cast_from(cls, items):
        return cls(Get(cls, items))

    @classmethod
    def expect(cls, spec):
        if typing.get_args(spec):
            spec = typing.get_args(spec)

        if len(spec) == 2 and spec[1] is Ellipsis:
            class _Tuple(cls):
                __spec__ = spec

            return _Tuple
        else:
            class _Tuple(cls):
                __spec__ = spec

                def __len__(self):
                    return len(self.__spec__)

                def __iter__(self):
                    return (self[i] for i in range(len(self)))

                def __reversed__(self):
                    return cls(tuple(self[i] for i in reversed(range(len(self)))))

            return _Tuple

    @classmethod
    def concatenate(cls, l, r):
        return cls(Post(uri(cls).append("concatenate"), {'l': l, 'r': r}))

    @classmethod
    def range(cls, range):
        """
        Construct a new :class:`Tuple` of :class:`Number` s in the given `range`.

        `range` can be a positive :class:`Number`, `(start, stop)`, or `(start, stop, step)`
        """

        from .scalar.number import Number
        return cls.expect(typing.Tuple[Number, ...])(Get(uri(cls) + "/range", range))

    def __new__(cls, form):
        if hasattr(form, "__iter__"):
            spec = tuple(type(s) if isinstance(s, State) else State for s in form)
            return State.__new__(cls.expect(spec))
        else:
            return State.__new__(cls)

    def __init__(self, form):
        while isinstance(form, Tuple):
            form = form_of(form)

        return State.__init__(self, form)

    def __add__(self, other):
        return self.concatenate(self, other)

    def __eq__(self, other):
        return self.eq(other)

    def __json__(self):
        return to_json(form_of(self))

    def __getitem__(self, i):
        if i is None:
            raise ValueError(f"invalid tuple index: {i}")

        spec = typing.get_args(self.__spec__) if typing.get_args(self.__spec__) else self.__spec__
        rtype = spec[0] if len(spec) == 2 and spec[1] is Ellipsis else State

        if isinstance(i, slice):
            if i.step is not None:
                raise NotImplementedError(f"slice with step: {i}")

            start = deref(i.start)
            stop = deref(i.stop)

            if len(spec) != 2 or (len(spec) == 2 and spec[1] is not Ellipsis):
                # the contents may be literal, so compute the slice now if possible
                if hasattr(form_of(self), "__getitem__"):
                    if is_literal(start) and is_literal(stop):
                        start = _index_of(start, len(self), 0)
                        stop = _index_of(stop, len(self), len(self))
                        return self.__class__([self[i] for i in range(start, stop)])

            return self._get("", Range.from_slice(i), self.__class__.expect(typing.Tuple[rtype, ...]))

        if isinstance(i, int):
            if len(spec) == 2 and spec[1] is Ellipsis:
                rtype = spec[0]
            elif i >= len(spec):
                raise IndexError(f"index {i} is out of bounds for {self}")
            else:
                rtype = spec[i]

            if hasattr(form_of(self), "__getitem__"):
                item = form_of(self)[i]
                if uri(self) == uri(self.__class__):
                    return item
                else:
                    return get_ref(item, uri(self).append(i))

        return self._get("", i, rtype)

    def __ne__(self, other):
        return self.ne(other)

    def __ref__(self, name):
        return self.__class__(form=TupleRef(self, name))

    def contains(self, item):
        from .scalar.number import Bool
        return self._get("contains", item, Bool)

    def eq(self, other):
        """Return a `Bool` indicating whether all elements in this `Tuple` equal those in the given `other`."""

        if same_as(self, other):
            return True

        from .scalar.number import Bool
        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """Return a `Bool` indicating whether the elements in this `Tuple` do not equal those in the given `other`."""

        return self.eq(other).logical_not()

    def len(self):
        """Return the number of elements in this `Tuple`."""

        from .scalar.number import UInt
        return self._get("len", rtype=UInt)

    def unpack(self, length):
        """A Python convenience method which yields an iterator over the first `length` elements in this `Tuple`."""

        yield from (self[i] for i in range(length))

    def zip(self, other):
        """Construct a new `Tuple` of 2-tuples of the form `(self[i], other[i]) for i in self.len()`."""

        return self._get("zip", other, type(self))


class TupleRef(StateRef):
    def __getitem__(self, i):
        return self.state[i]


def _index_of(i, length, default):
    if i is None:
        idx = default
    elif i < 0:
        idx = length + i
    else:
        idx = i

    return idx
