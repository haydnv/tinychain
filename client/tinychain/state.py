"""TinyChain `State` s, like `Map`, `Tuple`, and `Op`."""

import inspect

from tinychain.util import *


class State(object):
    """
    A TinyChain state, such as a `Chain` or `Op` or `Value`.

    Do not subclass `State` directly. Use a more specific type instead.
    """

    __uri__ = URI("/state")

    def __init__(self, form):
        from tinychain import reflect

        self.__form__ = form

        if isinstance(form, URI):
            self.__uri__ = form
        elif reflect.is_ref(form):
            self.__uri__ = uri(form)

        reflect.meta.gen_headers(self)

    def __dbg__(self):
        return [form_of(self)]

    def __eq__(self, _other):
        raise NotImplementedError("State does not support equality; use a more specific type")

    def __json__(self):
        from tinychain import reflect

        form = form_of(self)

        if reflect.is_ref(form):
            return to_json(form)
        else:
            return {str(uri(self)): [to_json(form)]}

    def __id__(self):
        return hex_id(form_of(self))

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
        from tinychain.new_state.ref import MethodSubject, Get

        subject = MethodSubject(self, name)
        op_ref = Get(subject, key)
        rtype = State if rtype is None or not issubclass(rtype, State) else rtype
        return rtype(op_ref)

    def _put(self, name, key=None, value=None):
        from tinychain.new_state.ref import MethodSubject, Put
        from tinychain.new_state.value import Nil

        subject = MethodSubject(self, name)
        return Nil(Put(subject, key, value))

    def _post(self, name, params, rtype):
        from tinychain.new_state.ref import MethodSubject, Post

        subject = MethodSubject(self, name)
        op_ref = Post(subject, params)
        rtype = State if rtype is None or not issubclass(rtype, State) else rtype
        return rtype(op_ref)

    def _delete(self, name, key=None):
        from tinychain.new_state.ref import MethodSubject, Delete
        from tinychain.new_state.value import Nil

        subject = MethodSubject(self, name)
        return Nil(Delete(subject, key))

    def cast(self, dtype):
        """Attempt to cast this `State` into the given `dtype`."""

        # TODO: allow casting to a type known only at run-time
        if not inspect.isclass(dtype) or not issubclass(dtype, State):
            raise NotImplementedError("dtype to cast into must be known at compile-time")

        from tinychain.new_state.ref import Get
        return dtype(Get(dtype, self))

    def copy(self):
        """Create a new `State` by copying this one."""

        return self._get("copy", rtype=self.__class__)

    def dtype(self):
        """Return the native base :class:`Class` of this `State`."""
        return self._get("class", rtype=Class)

    def hash(self):
        """Return the SHA256 hash of this `State` as an :class:`Id`."""

        from tinychain.new_state.value import Id
        return self._get("hash", rtype=Id)

    def is_none(self):
        """Return `Bool(true)` if this `State` is :class:`Nil`."""

        from tinychain.new_state.value import Bool
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
        if len(args) == 1:
            [form] = args
            if kwargs:
                raise ValueError(f"Map accepts a form or kwargs, not both (got {args}, {kwargs})")

            if isinstance(form, dict):
                return State.__init__(self, TypeForm(None, form))
            elif hasattr(form, "__form__") and isinstance(form_of(form), TypeForm):
                return State.__init__(self, form_of(form))
            else:
                return State.__init__(self, form)
        elif len(args) == 0:
            return State.__init__(self, TypeForm(None, kwargs))
        else:
            raise ValueError(f"Map accepts exactly one form argument (got {args})")

    def __eq__(self, other):
        return self.eq(other)

    def __getitem__(self, key):
        form = form_of(self)
        if isinstance(form, TypeForm):
            if key not in form:
                raise KeyError(f"Map with keys {set(form.keys())} does not contain any entry for {key}")

            rtype = type(form[key])
            rtype = rtype if issubclass(rtype, State) else State
            if hasattr(form[key], "__ref__"):
                return rtype(get_ref(form[key], f"{uri(self)}/{key}"))
        else:
            rtype = State

        return self._get("", key, rtype)

    def __json__(self):
        return to_json(form_of(self))

    def __ne__(self, other):
        return self.ne(other)

    def __ref__(self, name):
        form = form_of(self)
        if isinstance(form, TypeForm):
            type_data = {k: get_ref(form[k], f"{name}/{k}") for k in form}
            ref_form = TypeForm(URI(name), type_data)
            return self.__class__(ref_form)
        elif hasattr(form, "__ref__"):
            return self.__class__(get_ref(form, name))
        else:
            return self.__class__(URI(name))

    def eq(self, other):
        """Return a `Bool` indicating whether all the keys and values in this map are equal to the given `other`."""

        from tinychain.new_state.value import Bool
        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """Return a `Bool` indicating whether the keys and values in this `Map` are not equal to the given `other`."""

        return self.eq(other).logical_not()

    def len(self):
        """Return the number of elements in this `Map`."""

        from tinychain.new_state.value import UInt
        return self._get("len", rtype=UInt)


class Tuple(State):
    """A tuple of `State` s."""

    __uri__ = uri(State) + "/tuple"

    def __init__(self, form):
        if hasattr(form, "__form__") and isinstance(form_of(form), TypeForm):
            return State.__init__(self, form_of(form))
        else:
            form = TypeForm(None, form) if hasattr(form, "__iter__") else form
            return State.__init__(self, form)

    def __add__(self, other):
        return self.extend(other)

    def __eq__(self, other):
        return self.eq(other)

    def __json__(self):
        return to_json(form_of(self))

    def __getitem__(self, key):
        form = form_of(self)
        if isinstance(form, TypeForm):
            rtype = type(form[key])
            rtype = rtype if issubclass(rtype, State) else State
            if hasattr(form[key], "__ref__"):
                return rtype(get_ref(form[key], f"{uri(self)}/{key}"))
        else:
            rtype = State

        return self._get("", key, rtype)

    def __ne__(self, other):
        return self.ne(other)

    def __ref__(self, name):
        form = form_of(self)
        if isinstance(form, TypeForm):
            type_data = [get_ref(v, f"{name}/{i}") for i, v in enumerate(form)]
            ref_form = TypeForm(URI(name), type_data)
            return self.__class__(ref_form)
        elif hasattr(form, "__ref__"):
            return self.__class__(get_ref(form, name))
        else:
            return self.__class__(URI(name))

    def eq(self, other):
        """Return a `Bool` indicating whether all elements in this `Tuple` equal those in the given `other`."""

        from tinychain.new_state.value import Bool
        return self._post("eq", {"eq": other}, Bool)

    def ne(self, other):
        """Return a `Bool` indicating whether the elements in this `Tuple` do not equal those in the given `other`."""

        return self.eq(other).logical_not()

    def extend(self, other):
        """Construct a new `Tuple` which is the concatenation of this `Tuple` and the given `other`."""

        return self._get("extend", other, Tuple)

    def len(self):
        """Return the number of elements in this `Tuple`."""

        from tinychain.new_state.value import UInt
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


# Scalar types

class Scalar(State):
    """
    An immutable `State` which always resides entirely in the host's memory.

    Do not subclass `Scalar` directly. Use :class:`Value` instead.
    """

    __uri__ = uri(State) + "/scalar"

    def __json__(self):
        return to_json(form_of(self))


# User-defined object types

class Class(State):
    """A TinyChain class (possibly a user-defined class)."""

    __uri__ = uri(State) + "/object/class"

    def __call__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Class.__call__ accepts args or kwargs but not both")

        subject = ref.MethodSubject(self)
        if args:
            return ref.Get(subject, args)
        else:
            return ref.Get(subject, kwargs)


class Instance(State):
    """An instance of a user-defined :class:`Class`."""

    __uri__ = uri(State) + "/object/instance"

    def copy(self):
        raise NotImplementedError("abstract method")


# Private helper classes

class TypeForm(object):
    def __init__(self, form, type_data):
        from tinychain import reflect

        while isinstance(type_data, TypeForm):
            if form is None:
                form = type_data.__form__

            type_data = type_data.type_data

        if reflect.is_ref(form):
            self.__uri__ = uri(form)

        self.__form__ = form
        self.type_data = type_data

    def __getitem__(self, key):
        return self.type_data[key]

    def __iter__(self):
        return iter(self.type_data)

    def __ns__(self, cxt):
        if form_of(self) is not None:
            deanonymize(form_of(self), cxt)

        deanonymize(self.type_data, cxt)

    def __repr__(self):
        return f"State form {self.__form__} with associated type data {self.type_data}"

    def __json__(self):
        form = form_of(self)
        return to_json(self.type_data if form is None else form)

    def keys(self):
        return self.type_data.keys()
