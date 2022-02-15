import typing

from tinychain.util import deanonymize, form_of, get_ref, to_json, uri, URI

from . import State
from .value import Bool, Id, UInt


class Map(State):
    """A key-value map whose keys are `Id`s and whose values are `State` s."""

    __uri__ = uri(State) + "/map"
    __spec__ = dict[Id, State]

    @staticmethod
    def _parse_args(args, kwargs):
        form = None
        spec = None

        if args and "form" in kwargs:
            raise ValueError(f"Map got duplicate arguments for 'form': {args[0]}, {kwargs['form']}")

        if len(args) > 1 and "spec" in kwargs:
            raise ValueError(f"Map got duplicate arguments for 'form': {args[1]}, {kwargs['form']}")

        if len(args) > 2:
            raise ValueError(f"Map.__init__ got unexpected arguments {args} {kwargs}")

        if "form" in kwargs:
            form = kwargs["form"]
            del kwargs["form"]
        elif len(args) >= 1:
            form = args[0]

        if len(args) > 1:
            spec = args[1]

        if "spec" in kwargs:
            spec = kwargs["spec"]
            del kwargs["spec"]

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
        assert self.__form__ == form

    def __eq__(self, other):
        return self.eq(other)

    def __getitem__(self, key):
        if hasattr(form_of(self), key):
            return form_of(self)[key]

        if isinstance(self.__spec__, dict):
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
        if hasattr(form_of(self), "__ref__"):
            r = self.__class__(form=get_ref(form_of(self), name))
        else:
            r = self.__class__(form=URI(name))

        assert r.__spec__ == self.__spec__
        return r

    def eq(self, other):
        """Return a `Bool` indicating whether all the keys and values in this map are equal to the given `other`."""

        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """Return a `Bool` indicating whether the keys and values in this `Map` are not equal to the given `other`."""

        return self.eq(other).logical_not()

    def len(self):
        """Return the number of elements in this `Map`."""

        return self._get("len", rtype=UInt)


class Tuple(State):
    """A tuple of `State` s."""

    __uri__ = uri(State) + "/tuple"
    __spec__ = (State, ...)

    @classmethod
    def expect(cls, spec):
        if typing.get_args(spec):
            spec = typing.get_args(spec)

        class _Tuple(cls):
            __spec__ = spec

        return _Tuple

    def __new__(cls, form):
        if isinstance(form, Tuple):
            return State.__new__(cls)
        elif hasattr(form, "__iter__"):
            spec = tuple(type(s) if isinstance(s, State) else State for s in form)
            return State.__new__(cls.expect(spec))
        else:
            return State.__new__(cls)

    def __init__(self, form):
        if isinstance(form, Tuple):
            form = form_of(form)

        return State.__init__(self, form)

    def __add__(self, other):
        return self.extend(other)

    def __eq__(self, other):
        return self.eq(other)

    def __json__(self):
        return to_json(form_of(self))

    def __getitem__(self, i):
        if len(self.__spec__) == 2 and self.__spec__[1] is Ellipsis:
            rtype = self.__spec__[0]
        elif i < len(self.__spec__):
            rtype = self.__spec__[i]
        else:
            raise IndexError(f"index {i} out of bounds for {self}")

        return self._get("", i, rtype)

    def __len__(self):
        if len(self.__spec__) == 2 and self.__spec__[1] is Ellipsis:
            raise RuntimeError(f"the length of {self} is not known at compile-time")
        else:
            return len(self.__spec__)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __ne__(self, other):
        return self.ne(other)

    def __ref__(self, name):
        if hasattr(form_of(self), "__ref__"):
            return self.__class__(form=get_ref(form_of(self), name))
        else:
            return self.__class__(form=URI(name))

    def eq(self, other):
        """Return a `Bool` indicating whether all elements in this `Tuple` equal those in the given `other`."""

        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """Return a `Bool` indicating whether the elements in this `Tuple` do not equal those in the given `other`."""

        return self.eq(other).logical_not()

    def extend(self, other):
        """Construct a new `Tuple` which is the concatenation of this `Tuple` and the given `other`."""

        return self._get("extend", other, Tuple)

    def len(self):
        """Return the number of elements in this `Tuple`."""

        return self._get("len", rtype=UInt)

    def fold(self, item_name, initial_state, op):
        """Iterate over the elements in this `Tuple` with the given `op`, accumulating the results.
        `op` must be a POST Op. The stream item to handle will be passed with the given `item_name` as its name.
        """

        rtype = type(initial_state) if isinstance(initial_state, State) else State
        return self._post("fold", {"item_name": item_name, "value": initial_state, "op": op}, rtype)

    def map(self, op):
        """Construct a new `Tuple` by mapping the elements in this `Tuple` with the given `op`."""

        rtype = op.rtype if hasattr(op, "rtype") else State
        return self._post("map", {"op": op}, rtype)

    def unpack(self, length):
        """A Python convenience method which yields an iterator over the first `length` elements in this `Tuple`."""

        yield from (self[i] for i in range(length))

    def zip(self, other):
        """Construct a new `Tuple` of 2-tuples of the form `(self[i], other[i]) for i in self.len()`."""

        return self._get("zip", other, Tuple)
