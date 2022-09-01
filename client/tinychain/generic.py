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
            spec = self.__orig_class__.__args__[0]
            if typing.get_origin(spec) is dict:
                _id_type, rtype = typing.get_args(spec)
            else:
                rtype = resolve_class(spec)
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
    """A named reference to a :class:`Map`."""

    def __getitem__(self, key):
        return self.state[key]


class Tuple(State, Functional, typing.Generic[T]):
    """A tuple of :class:`State` s."""

    __uri__ = URI(State) + "/tuple"

    def __new__(cls, form):
        if hasattr(form, "__iter__"):
            class _Tuple(cls):
                def __iter__(self):
                    return (self[i] for i in range(len(self)))

                def __len__(self):
                    return len(form)

                def __reversed__(self):
                    return (self[i] for i in reversed(range(len(self))))

            return State.__new__(_Tuple)
        else:
            return State.__new__(cls)

    @classmethod
    def cast_from(cls, items):
        return cls(Get(cls, items))

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
        return Tuple[Number](form=Get(URI(cls) + "/range", range))

    def __init__(self, form):
        form = tuple(form) if hasattr(form, "__iter__") else form
        return State.__init__(self, form)

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

        if isinstance(i, Range):
            i = i.to_slice()

        if isinstance(i, slice):
            if i.step is not None:
                raise NotImplementedError(f"slice with step: {i}")

            start = deref(i.start)
            stop = deref(i.stop)

            # the contents may be literal, so compute the slice now if possible
            if hasattr(form_of(self), "__getitem__"):
                if is_literal(start) and is_literal(stop):
                    length = len(deref(self))
                    start = _index_of(start, length, 0)
                    stop = _index_of(stop, length, length)
                    return Tuple([self[i] for i in range(start, stop)])

            return self._get("", Range.from_slice(i), Tuple)

        if isinstance(deref(i), int):
            if hasattr(form_of(self), "__getitem__"):
                item = form_of(self)[deref(i)]
                if self.__uri__ == type(self).__uri__:
                    return item
                else:
                    return get_ref(item, self.__uri__.append(i))

        if hasattr(self, "__orig_class__"):
            (spec,) = typing.get_args(self.__orig_class__)
            if typing.get_origin(spec) is tuple:
                spec = typing.get_args(spec)
                if len(spec) == 2 and spec[1] is Ellipsis:
                    rtype = spec[0]
                elif isinstance(i, slice):
                    rtype = typing.Tuple[spec[i]]
                elif isinstance(deref(i), int):
                    rtype = spec[deref(i)]
                else:
                    rtype = gcs([resolve_class(t) for t in spec])
            else:
                rtype = resolve_class(spec)
        else:
            rtype = State

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
    """A named reference to a :class:`Tuple`."""

    def __getitem__(self, i):
        return self.state[i]


def autobox(state):
    """
    Given a Python literal, box it in the appropriate type of `State`.

    For example, `autobox(1)` will return `Int(1)`, `autobox({'a': 3.14})` will return a `Map[Float]`, etc.
    """

    if isinstance(state, State) or (inspect.isclass(state) and issubclass(state, State)):
        return state

    if isinstance(state, bool):
        from .scalar.number import Bool
        return Bool(state)
    elif isinstance(state, int):
        from .scalar.number import Int
        return Int(state)
    elif isinstance(state, dict):
        state = {k: autobox(v) for k, v in state.items()}
        vtype = gcs(*[type(v) for v in state.values()]) if all(isinstance(v, State) for v in state.values()) else State
        return Map[vtype](state)
    elif isinstance(state, (list, tuple)):
        state = tuple(autobox(item) for item in state)
        dtype = gcs(*[type(item) for item in state]) if all(isinstance(item, State) for item in state) else State
        return Tuple[dtype](state)
    elif isinstance(state, str):
        from .scalar.value import String
        return String(state)
    elif isinstance(state, float):
        from .scalar.number import Float
        return Float(state)
    elif isinstance(state, complex):
        from .scalar.number import Complex
        return Complex(state)

    from .scalar.ref import Ref
    if isinstance(state, Ref):
        return State(form=state)

    if isinstance(state, Interface):
        return resolve_interface(type(state))(form=state)

    return state


def gcs(*types):
    """Get the greatest common superclass of a list of types"""

    assert all(isinstance(t, type) for t in types)
    assert not any(t is type for t in types)

    if not types:
        return State

    classes = [t.mro() for t in types]
    for x in classes[0]:
        if all(x in mro for mro in classes):
            return x


def resolve_class(type_hint):
    """Given a generic type hint, attempt to resolve it to a callable class constructor."""

    if inspect.isclass(type_hint):
        if issubclass(type_hint, State):
            return type_hint
        elif issubclass(type_hint, Interface):
            return resolve_interface(type_hint)
    elif callable(type_hint):
        # assume this is a generic type alias which will return an instance of the correct type when called
        return type_hint

    if type_hint is typing.Any:
        return State
    else:
        raise NotImplementedError(f"resolve type hint {type_hint}")


def resolve_interface(cls):
    """Construct a subclass of :class:`State` which implements the given :class:`Interface`."""

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
