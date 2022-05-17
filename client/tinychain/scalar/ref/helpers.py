import inspect
import logging

from ...uri import uri, URI
from ...context import to_json

from .base import Case, If, MethodSubject, Ref


def args(state_or_ref):
    """
    Return the immediate references needed to resolve the given `state_or_ref`.

    This function is not recursive. Use `depends_on` to get all compile-time dependencies of `state_or_ref`.
    """

    if form_of(state_or_ref) is not state_or_ref:
        return args(form_of(state_or_ref))

    return state_or_ref.__args__() if hasattr(state_or_ref, "__args__") else []


def depends_on(state_or_ref):
    """Return the set of all compile-time dependencies of the given `state_or_ref`"""

    if is_literal(state_or_ref):
        return set()

    if form_of(state_or_ref) is not state_or_ref:
        return depends_on(form_of(state_or_ref))

    if independent(state_or_ref):
        return set() if is_literal(state_or_ref) else set([state_or_ref])

    deps = set()

    if isinstance(state_or_ref, Ref):
        for dep in args(state_or_ref):
            deps.update(depends_on(dep))
    elif isinstance(state_or_ref, (list, tuple)):
        for dep in state_or_ref:
            deps.update(depends_on(dep))
    elif isinstance(state_or_ref, dict):
        for dep in state_or_ref.values():
            deps.update(depends_on(dep))

    return deps


def deref(state):
    """Return the :class:`Ref`, :class:`URI`, or constant which will be used to resolve the given :class:`State`."""

    from .base import FlowControl, Op as OpRef

    if isinstance(state, (FlowControl, OpRef)) or callable(state) or inspect.isclass(state):
        return state

    if form_of(state) is not state:
        return deref(form_of(state))
    elif isinstance(state, Ref) and hasattr(state, "state"):
        return deref(state.state)
    elif isinstance(state, MethodSubject):
        return deref(state.subject)
    else:
        return state


def form_of(state):
    """Return the form of the given `state`."""

    if not hasattr(state, "__form__"):
        return state

    if callable(state.__form__) and not inspect.isclass(state.__form__):
        form = state.__form__()
    else:
        form = state.__form__

    return form


def get_ref(state, name):
    """Return a named reference to the given `state`."""

    name = URI(name)

    if inspect.isclass(state):
        def ctr(*args, **kwargs):
            return state(*args, **kwargs)

        ctr.__uri__ = name
        ctr.__json__ = lambda: to_json(uri(ctr))
        return ctr
    elif hasattr(state, "__ref__"):
        return state.__ref__(name)
    elif isinstance(state, dict):
        return {k: get_ref(v, name.append(k)) for k, v in state.items()}
    elif isinstance(state, (list, tuple)):
        return tuple(get_ref(item, name.append(i)) for i, item in enumerate(state))
    else:
        logging.debug(f"{state} has no __ref__ method")
        return state


def hex_id(state_or_ref):
    """
    Return a unique hexadecimal string identifying the given `state_or_ref` based on its memory address.

    This is similar to Python's built-in `id` function but has the advantage that it will still produce the correct
    unique ID even for wrapper types. For example:

    ```python
    x = 1
    n = Number(x)
    assert hex_id(n) == hex_id(x)  # this won't work with the built-in `id` function
    ```
    """

    if hasattr(state_or_ref, "__id__"):
        return state_or_ref.__id__()

    return format(id(state_or_ref), 'x')


def independent(state_or_ref):
    """Return `True` if the given `state_or_ref` does not depend on any unresolved references."""

    if form_of(state_or_ref) is not state_or_ref:
        return independent(form_of(state_or_ref))

    from ... import reflect

    if isinstance(state_or_ref, (reflect.method.Method, reflect.op.Op)) or inspect.isclass(state_or_ref):
        return True

    if isinstance(state_or_ref, dict):
        return all(independent(value) for value in state_or_ref.values())
    elif isinstance(state_or_ref, (list, tuple)):
        return all(independent(item) for item in state_or_ref)
    elif isinstance(state_or_ref, Ref):
        return not args(state_or_ref)
    else:
        return True


def is_literal(state):
    from ...generic import Map, Tuple

    if isinstance(state, (list, tuple)):
        return all(is_literal(item) for item in state)
    elif isinstance(state, dict):
        return all(is_literal(value) for value in state.values())
    elif isinstance(state, slice):
        return is_literal(state.start) and is_literal(state.stop)
    elif isinstance(state, (bool, float, int, str)):
        return True
    elif state is None:
        return True
    elif isinstance(state, (Map, Tuple)):
        return is_literal(form_of(state))

    return False


def is_conditional(state):
    from ...state import State

    if isinstance(state, State):
        return is_conditional(form_of(state))
    elif isinstance(state, dict):
        return any(is_conditional(value) for value in state.values())
    elif isinstance(state, (list, tuple)):
        return any(is_conditional(item) for item in state)

    return isinstance(state, (Case, If))


def is_none(state):
    from ..value import Nil

    return state is None or state == Nil


def is_op(fn):
    from ...reflect.method import Method
    from ...reflect.op import Op

    if isinstance(fn, (Method, Op)):
        return True
    elif hasattr(fn, "__form__"):
        return is_op(form_of(fn))
    elif isinstance(fn, (list, tuple)):
        return any(is_op(item) for item in fn)
    elif isinstance(fn, dict):
        return any(is_op(fn[k]) for k in fn)
    else:
        return False


def is_op_ref(state_or_ref, allow_literals=True):
    """Return `True` if `state_or_ref` is a reference to an `Op`, otherwise `False`."""

    from .base import After, Case, If, Op

    if allow_literals and is_literal(state_or_ref):
        return False
    elif isinstance(state_or_ref, (Op, After, If, Case)):
        return True
    elif uri(state_or_ref) and uri(type(state_or_ref)) and uri(state_or_ref) >= uri(type(state_or_ref)):
        return True
    elif form_of(state_or_ref) is not state_or_ref:
        return is_op_ref(form_of(state_or_ref), allow_literals)
    elif isinstance(state_or_ref, (list, tuple)):
        return any(is_op_ref(item) for item in state_or_ref)
    elif isinstance(state_or_ref, dict):
        return any(is_op_ref(state_or_ref[k]) for k in state_or_ref)
    else:
        return False


def is_write_op_ref(fn):
    """Return `True` if `state_or_ref` is a reference to a `Put` or `Delete` op, otherwise `False`."""

    from .base import Delete, Put

    if isinstance(fn, (Delete, Put)):
        return True
    elif hasattr(fn, "__form__"):
        return is_write_op_ref(form_of(fn))
    elif isinstance(fn, (list, tuple)):
        return any(is_write_op_ref(item) for item in fn)
    elif isinstance(fn, dict):
        return any(is_write_op_ref(fn[k]) for k in fn)
    else:
        return False


def is_ref(state):
    if isinstance(state, (Ref, URI, MethodSubject)):
        return True
    elif hasattr(state, "__form__"):
        return is_ref(form_of(state))
    elif isinstance(state, (list, tuple)):
        return any(is_ref(item) for item in state)
    elif isinstance(state, dict):
        return any(is_ref(state[k]) for k in state)
    else:
        return False


def reference(context, state, name_hint):
    """Create a reference to `state` in `context` using its `hex_id`"""

    i = 0
    name = name_hint
    while name in context and not same_as(getattr(context, name), state):
        name = f"{name_hint}_{i}"
        i += 1

    if name not in context:
        logging.debug(f"assigned name {name} to {state} in {context}")
        setattr(context, name, state)

    return getattr(context, name)


def same_as(a, b):
    """Return `True` if `a` is logically equivalent to `b`, otherwise `False`."""

    if a is b:
        return True

    a = deref(a)
    b = deref(b)

    if type(a) is type(b) and hasattr(a, "__same__"):
        return a.__same__(b)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_as(a_item, b_item) for a_item, b_item in zip(a, b))
    elif isinstance(a, dict) and isinstance(b, dict):
        return set(a.keys()) == set(b.keys()) and all(same_as(a[k], b[k]) for k in a)
    elif is_literal(a) and is_literal(b):
        return a == b
    else:
        return a is b
