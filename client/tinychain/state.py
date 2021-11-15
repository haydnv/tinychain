"""TinyChain `State` s, such as `Map`, `Tuple`, and `Op`."""

from tinychain import ref, reflect
from tinychain.util import *


class State(object):
    """
    A TinyChain state, such as a `Chain` or `Op` or `Value`.

    Do not subclass `State` directly. Use a more specific type instead.
    """

    __uri__ = URI("/state")

    def __init__(self, form):
        self.__form__ = form

        if isinstance(form, URI):
            self.__uri__ = form

        reflect.meta.gen_headers(self)

    def __deps__(self):
        return requires(form_of(self))

    def __json__(self):
        form = form_of(self)

        if reflect.is_ref(form):
            return to_json(form)
        else:
            return {str(uri(self)): [to_json(form)]}

    def __ns__(self, cxt):
        deanonymize(form_of(self), cxt)

        if isinstance(self.__form__, URI):
            self.__uri__ = self.__form__

    def __ref__(self, name):
        if hasattr(form_of(self), "__ref__"):
            return self.__class__(get_ref(form_of(self), name))
        else:
            return self.__class__(URI(name))

    def __repr__(self):
        return f"{self.__class__.__name__}({form_of(self)})"

    def _get(self, name, key=None, rtype=None):
        subject = ref.MethodSubject(self, name)
        op_ref = ref.Get(subject, key)
        rtype = State if rtype is None else rtype
        return rtype(op_ref)

    def _put(self, name, key=None, value=None):
        from .value import Nil

        subject = ref.MethodSubject(self, name)
        return Nil(ref.Put(subject, key, value))

    def _post(self, name, params, rtype):
        from .value import Nil

        subject = ref.MethodSubject(self, name)
        op_ref = ref.Post(subject, params)
        rtype = Nil if rtype is None else rtype
        return rtype(op_ref)

    def _delete(self, name, key=None):
        from .value import Nil

        subject = ref.MethodSubject(self, name)
        return Nil(ref.Delete(subject, key))

    def cast(self, dtype):
        """Attempt to cast this `State` into the given `dtype`."""

        return dtype(ref.Get(uri(dtype), self))

    def dtype(self):
        """Return the native base :class:`Class` of this `State`."""
        return self._get("class", rtype=Class)

    def is_none(self):
        """Return `Bool(true)` if this `State` is :class:`Nil`."""

        from .value import Bool
        return self._get("is_none", rtype=Bool)

    def is_some(self):
        """
        Return `Bool(true)` if this `State` is not :class:`Nil`.

        This is defined as `self.is_none().logical_not()`.
        """

        return self.is_none().logical_not()


class Map(State):
    """A key-value map whose keys are `Id`s and whose values are `State` s."""

    __uri__ = uri(State) + "/map"

    def __init__(self, *args, **kwargs):
        if args:
            if kwargs:
                raise ValueError(f"Map accepts a form or kwargs, not both (got {kwargs})")

            [form] = args
            State.__init__(self, form)
        else:
            State.__init__(self, kwargs)

    def __eq__(self, other):
        return self.eq(other)

    def __getitem__(self, key):
        form = form_of(self)
        if hasattr(form, "__getitem__"):
            rtype = type(form[key]) if issubclass(type(form[key]), State) else State
            return self._get("", key, rtype)
        else:
            return self._get("", key)

    def __json__(self):
        return to_json(form_of(self))

    def __ne__(self, other):
        return self.ne(other)

    def eq(self, other):
        """Return a `Bool` indicating whether all the keys and values in this map are equal to the given `other`."""

        from .value import Bool
        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """Return a `Bool` indicating whether the keys and values in this `Map` are not equal to the given `other`."""

        return self.eq(other).logical_not()

    def len(self):
        """Return the number of elements in this `Map`."""

        from .value import UInt
        return self._get("len", rtype=UInt)


class Tuple(State):
    """A tuple of `State` s."""

    __uri__ = uri(State) + "/tuple"

    def __add__(self, other):
        return self.extend(other)

    def __eq__(self, other):
        return self.eq(other)

    def __json__(self):
        return to_json(form_of(self))

    def __getitem__(self, i):
        form = form_of(self)
        if hasattr(form, "__iter__"):
            rtype = type(form[i]) if issubclass(type(form[i]), State) else State
            return self._get("", i, rtype)
        else:
            return self._get("", i)

    def __ne__(self, other):
        return self.ne(other)

    def eq(self, other):
        """Return a `Bool` indicating whether all elements in this `Tuple` equal those in the given `other`."""

        from .value import Bool
        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """Return a `Bool` indicating whether the elements in this `Tuple` do not equal those in the given `other`."""

        return self.eq(other).logical_not()

    def extend(self, other):
        """Construct a new `Tuple` which is the concatenation of this `Tuple` and the given `other`."""

        return self._get("extend", other, Tuple)

    def len(self):
        """Return the number of elements in this `Tuple`."""

        from .value import UInt
        return self._get("len", rtype=UInt)

    # TODO: update this method signature to match `Stream.fold`
    def fold(self, initial_state, op):
        """Iterate over the elements in this `Tuple` with the given `op`, accumulating the results."""

        rtype = type(initial_state) if issubclass(type(initial_state), State) else State
        return self._post("fold", {"value": initial_state, "op": op}, rtype)

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


# Scalar types

class Scalar(State):
    """
    An immutable `State` which always resides entirely in the host's memory.

    Do not subclass `Scalar` directly. Use :class:`Value` instead.
    """

    __uri__ = uri(State) + "/scalar"

    def __json__(self):
        return to_json(form_of(self))


# A stream of `State` s

class Stream(State):
    """A stream of states which supports functional methods like `fold` and `map`."""

    __uri__ = uri(State) + "/stream"

    @classmethod
    def range(cls, range):
        """Return a stream of numbers in the given `range`.

        `range` can be a positive number, a 2-Tuple like `(start, stop)`, or a 3-Tuple like `(start, stop, step)`
        """

        return cls(ref.Get(uri(cls) + "/range", range))

    def aggregate(self):
        return self._get("aggregate", rtype=Stream)

    def first(self):
        """Return the first item in this `Stream`, or `Nil` if the `Stream` is empty."""

        return self._get("first", rtype=Scalar)

    def for_each(self, op):
        """Run the given `op` for each item in this `Stream`, then return the last result.

        This is useful when you need to execute an `op` for its side-effects and not its return value.
        """

        rtype = op.rtype if hasattr(op, "rtype") else State
        return self._post("for_each", Map(op=op), rtype)

    def fold(self, item_name, initial_state, op):
        """Run the given `op` for each item in this `Stream` along with the previous result.

        `op` must be a POST Op. The stream item to handle will be passed with the given `item_key` as its name.
        """

        return self._post("fold", Map(item_name=item_name, value=initial_state, op=op), type(initial_state))

    def map(self, op):
        """Return a new `Stream` whose items are the results of running `op` on each item of this `Stream."""

        return self._post("map", Map(op=op), Stream)


# User-defined object types

class Class(State):
    """A user-defined TinyChain class."""

    __uri__ = uri(State) + "/object/class"


class Instance(State):
    """An instance of a user-defined :class:`Class`."""

    __uri__ = uri(State) + "/object/instance"

