import inspect

from ...interface import Interface
from ...uri import URI

from .base import Case, If, Ref


def args(ref):
    """Return the arguments needed to reconstruct the given `ref`."""

    if hasattr(ref, "__args__"):
        return ref.__args__()

    raise TypeError(f"not a reference: {ref} (type {type(ref)})")


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

    if isinstance(state, (FlowControl, OpRef)) or inspect.isclass(state):
        return state

    if isinstance(state, Ref) and hasattr(state, "state"):
        return deref(state.state)
    elif form_of(state) is not state:
        return deref(form_of(state))
    else:
        return state


def form_of(state):
    """Return the form of the given `state`."""

    if not hasattr(state, "__form__"):
        return state

    if callable(state.__form__) and not inspect.isclass(state.__form__) and not hasattr(state.__form__, "__form__"):
        form = state.__form__()
    else:
        form = state.__form__

    return form


def get_ref(state, name):
    """Return a named reference to the given `state`."""

    if not isinstance(name, URI):
        name = URI(name)

    if hasattr(state, "__ref__") and not inspect.isclass(state):
        return state.__ref__(name)
    elif isinstance(state, dict):
        return {k: get_ref(v, name.append(k)) for k, v in state.items()}
    elif isinstance(state, (list, tuple)):
        return tuple(get_ref(item, name.append(i)) for i, item in enumerate(state))
    elif isinstance(state, Interface):
        from ...state import State
        return type(f"{state.__class.__name__}State", (State, type(state)), {})(form=state)
    else:
        return state


def hex_id(state):
    """Return the memory address of the form of the given `state`"""

    if hasattr(state, "__id__"):
        return state.__id__()
    else:
        return id(state)


def independent(state_or_ref):
    """Return `True` if the given `state_or_ref` does not depend on any unresolved references."""

    if form_of(state_or_ref) is not state_or_ref:
        return independent(form_of(state_or_ref))

    from ... import reflect

    if isinstance(state_or_ref, (reflect.method.Method, reflect.op.Op)) or inspect.isclass(state_or_ref):
        return True

    if isinstance(state_or_ref, dict):
        return all(independent(value) and not is_op_ref(value) for value in state_or_ref.values())
    elif isinstance(state_or_ref, (list, tuple)):
        return all(independent(item) and not is_op_ref(item) for item in state_or_ref)
    elif isinstance(state_or_ref, Ref):
        return not is_op_ref(args(state_or_ref))
    else:
        return True


def is_literal(state):
    """Return `True` if the given `state` is a Python literal (or a type expectation wrapping a Python literal)."""

    from ...state import State

    if isinstance(state, (list, tuple)):
        return all(is_literal(item) for item in state)
    elif isinstance(state, dict):
        return all(is_literal(value) for value in state.values())
    elif isinstance(state, slice):
        return is_literal(state.start) and is_literal(state.stop)
    elif isinstance(state, (bool, complex, float, int, str)):
        return True
    elif state is None:
        return True
    elif isinstance(state, State):
        return is_literal(form_of(state))

    return False


def is_conditional(state):
    """Return `True` if the given `state` depends on a :class:`Case` or :class:`If` reference."""

    from ...state import State

    if isinstance(state, State):
        return is_conditional(form_of(state))
    elif isinstance(state, dict):
        return any(is_conditional(value) for value in state.values())
    elif isinstance(state, (list, tuple)):
        return any(is_conditional(item) for item in state)

    return isinstance(state, (Case, If))


def is_none(state):
    """Return `True` if the given `state` is `None` or `Nil`."""

    from ..value import Nil
    return state is None or type(state) is Nil


def is_op(fn):
    """Return `True` if the given `fn` is a reflected :class:`Op` or :class:`Method`."""

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
    elif hasattr(state_or_ref, "__uri__") and hasattr(type(state_or_ref), "__uri__") and state_or_ref.__uri__.startswith(type(state_or_ref).__uri__):
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
    """Return `True` if `state` depends on an unresolved reference."""

    if isinstance(state, (Ref, URI)):
        return True
    elif hasattr(state, "__form__"):
        if callable(state.__form__) and not hasattr(state.__form__, "__form__"):
            # special case: it's an Op and we don't want to call __form__
            return False

        return is_ref(form_of(state))
    elif isinstance(state, (list, tuple)):
        return any(is_ref(item) for item in state)
    elif isinstance(state, dict):
        return any(is_ref(state[k]) for k in state)
    else:
        return False


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
