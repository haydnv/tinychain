from tinychain.util import deanonymize, form_of, get_ref, to_json, uri, URI

from . import State


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
