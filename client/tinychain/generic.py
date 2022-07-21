import inspect
import typing

from .interface import Functional, Interface
from .json import to_json
from .scalar.bound import Range
from .scalar.ref import deref, form_of, get_ref, is_literal, same_as, Get, Post
from .state import State, StateRef
from .uri import URI

T = typing.TypeVar("T", bound=State)


# TODO: implement `Functional` for `Map`
class Map(State, typing.Generic[T]):
    """A key-value map whose keys are :class:`Id` s and whose values are :class: `State` s."""

    __uri__ = URI(State) + "/map"

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise RuntimeError(f"Map takes a form or kwargs, not both (got {args} and {kwargs}")
        elif kwargs:
            if "form" in kwargs:
                if len(kwargs) > 1:
                    raise KeyError(f"the name 'form' is reserved (Map got kwargs {kwargs})")

                form = kwargs["form"]
            else:
                form = kwargs
        elif len(args) == 1:
            [form] = args
        else:
            raise RuntimeError(f"cannot construct a Map without a form")

        State.__init__(self, form)

    def __eq__(self, other):
        return self.eq(other)

    def __getitem__(self, key):
        if hasattr(form_of(self), "__getitem__"):
            if self.__uri__ == type(self).__uri__:
                return form_of(self)[key]
            else:
                return get_ref(form_of(self)[key], self.__uri__.append(key))

        if hasattr(self, "__orig_class__"):
            rtype = resolve_class(self.__orig_class__.__args__[0])
        else:
            rtype = State

        return self._get("", key, rtype)

    def __hash__(self):
        return State.__hash__(self)

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
    """A tuple of :class:`State` s."""

    __uri__ = URI(State) + "/tuple"
    __spec__ = (State, ...)  # TODO: delete

    @classmethod
    def cast_from(cls, items):
        return cls(Get(cls, items))

    @classmethod
    def expect(cls, spec):  # TODO: delete
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
        return cls(Post(URI(cls).append("concatenate"), {'l': l, 'r': r}))

    @classmethod
    def range(cls, range):
        """
        Construct a new :class:`Tuple` of :class:`Number` s in the given `range`.

        `range` can be a positive :class:`Number`, `(start, stop)`, or `(start, stop, step)`
        """

        from .scalar.number import Number
        return cls.expect(typing.Tuple[Number, ...])(Get(URI(cls) + "/range", range))

    def __new__(cls, form):
        if hasattr(form, "__iter__"):
            spec = tuple(type(s) if isinstance(s, State) else State for s in form)
            return State.__new__(cls.expect(spec))
        else:
            return State.__new__(cls)

    def __init__(self, form):
        while isinstance(form, Tuple):
            form = form_of(form)

        return State.__init__(self, tuple(form) if isinstance(form, list) else form)

    def __add__(self, other):
        return Tuple.concatenate(self, other)

    def __eq__(self, other):
        return self.eq(other)

    def __hash__(self):
        return State.__hash__(self)

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
                if self.__uri__ == type(self).__uri__:
                    return item
                else:
                    return get_ref(item, self.__uri__.append(i))

        return self._get("", i, rtype)

    def __ne__(self, other):
        return self.ne(other)

    def __ref__(self, name):
        return self.__class__(form=TupleRef(self, name))

    def contains(self, item):
        """Return a :class:`Bool` indicating whether the given `item` is present in this :class:`Tuple`."""
        from .scalar.number import Bool
        return self._get("contains", item, Bool)

    def eq(self, other):
        """
        Return a :class:`Bool` indicating whether all elements in this :class:`Tuple` equal those in the given `other`.
        """

        if same_as(self, other):
            return True

        from .scalar.number import Bool
        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """
        Return a :class:`Bool` indicating whether the elements in this :class:`Tuple` do not equal those in `other`.

        This is implemented as `self.eq(other).logical_not()`.
        """

        return self.eq(other).logical_not()

    def len(self):
        """Return the number of elements in this :class:`Tuple`."""

        from .scalar.number import UInt
        return self._get("len", rtype=UInt)

    def unpack(self, length):
        """
        A Python convenience method which yields an iterator over the first `length` elements in this :class:`Tuple`.
        """

        yield from (self[i] for i in range(length))

    def zip(self, other):
        """Construct a new `Tuple` of 2-tuples of the form `(self[i], other[i]) for i in self.len()`."""

        return self._get("zip", other, type(self))


class TupleRef(StateRef):
    def __getitem__(self, i):
        return self.state[i]


def gcs(*types):
    """Get the greatest common superclass of a list of types"""

    classes = [t.mro() for t in types]
    for x in classes[0]:
        if all(x in mro for mro in classes):
            return x


def resolve_class(type_hint):
    if inspect.isclass(type_hint):
        if issubclass(type_hint, State):
            return type_hint
        elif issubclass(type_hint, Interface):
            return resolve_interface(type_hint)

    if type_hint is typing.Any:
        return State
    elif typing.get_origin(type_hint) is tuple:
        return Tuple.expect(type_hint)
    elif typing.get_origin(type_hint) is dict:
        return Map[type_hint]
    else:
        raise NotImplementedError(f"resolve type hint {type_hint}")


def resolve_interface(cls):
    assert inspect.isclass(cls)

    if issubclass(cls, Interface) and not issubclass(cls, State):
        return type(f"{cls.__name__}State", (State, cls), {})
    else:
        return cls


def _index_of(i, length, default):
    if i is None:
        idx = default
    elif i < 0:
        idx = length + i
    else:
        idx = i

    return idx
