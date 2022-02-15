import inspect

from pydoc import locate

from tinychain.util import deanonymize, form_of, get_ref, to_json, uri, URI

from .meta import gen_headers, header, Meta


def is_conditional(state):
    from tinychain.state.ref import Case, If
    from tinychain.state import State

    if isinstance(state, State):
        return is_conditional(form_of(state))

    return isinstance(state, Case) or isinstance(state, If)


def is_none(state):
    from tinychain.state.value import Nil

    return state is None or state == Nil


def is_op(fn):
    from .method import Method
    from .op import Op

    if isinstance(fn, Method) or isinstance(fn, Op):
        return True
    elif hasattr(fn, "__form__"):
        return is_op(form_of(fn))
    elif isinstance(fn, list) or isinstance(fn, tuple):
        return any(is_op(item) for item in fn)
    elif isinstance(fn, dict):
        return any(is_op(fn[k]) for k in fn)
    else:
        return False


def is_ref(state):
    from tinychain.state.ref import MethodSubject, Ref

    if isinstance(state, Ref) or isinstance(state, URI) or isinstance(state, MethodSubject):
        return True
    elif hasattr(state, "__form__"):
        return is_ref(form_of(state))
    elif isinstance(state, list) or isinstance(state, tuple):
        return any(is_ref(item) for item in state)
    elif isinstance(state, dict):
        return any(is_ref(state[k]) for k in state)
    else:
        return False


def resolve_class(subject, annotation, default):
    if annotation == inspect.Parameter.empty:
        return default
    elif inspect.isclass(annotation):
        return annotation

    classpath = f"{subject.__module__}.{annotation}"
    resolved = locate(classpath)
    if resolved is None:
        raise ValueError(f"unable to resolve class {classpath}")
    else:
        return resolved


def _get_rtype(fn, default_rtype):
    if is_op(fn):
        return fn.rtype

    rtype = default_rtype

    if inspect.isfunction(fn):
        annotation = inspect.signature(fn).return_annotation
        rtype = resolve_class(fn, annotation, rtype)

    return rtype
