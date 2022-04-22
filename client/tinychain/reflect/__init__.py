import inspect
import typing

from pydoc import locate

from ..util import form_of, URI

from .meta import Meta, MethodStub


def is_conditional(state):
    from ..scalar.ref import Case, If
    from ..state import State

    if isinstance(state, State):
        return is_conditional(form_of(state))
    elif isinstance(state, dict):
        return any(is_conditional(value) for value in state.values())
    elif isinstance(state, (list, tuple)):
        return any(is_conditional(item) for item in state)

    return isinstance(state, (Case, If))


def is_none(state):
    from ..scalar.value import Nil

    return state is None or state == Nil


def is_op(fn):
    from .method import Method
    from .op import Op

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


def is_ref(state):
    from ..scalar.ref import MethodSubject, Ref

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


def parse_args(sig, *args, **kwargs):
    params = {}

    i = 0
    for name, param in sig:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"variable positional arguments are not supported")
        elif param.kind == inspect.Parameter.POSITIONAL_ONLY:
            if i < len(args):
                params[name] = args[i]
                i += 1
            elif param.default is inspect.Parameter.empty:
                raise TypeError(f"missing required positional argument {name}")
            else:
                params[name] = param.default
        elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if i < len(args):
                params[name] = args[i]
                i += 1
            elif name in kwargs:
                params[name] = kwargs[name]
            elif param.default is inspect.Parameter.empty:
                raise TypeError(f"missing required argument {name}")
            else:
                params[name] = param.default
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            params.update(kwargs)

    return params


def resolve_class(subject, annotation, default):
    if annotation == inspect.Parameter.empty:
        return default
    elif typing.get_origin(annotation) is tuple:
        from ..generic import Tuple

        return Tuple.expect(annotation)
    elif typing.get_origin(annotation) is dict:
        from ..generic import Map

        return Map.expect(annotation)
    elif inspect.isclass(annotation):
        return _resolve_interface(annotation)

    classpath = f"{subject.__module__}.{annotation}"
    resolved = locate(classpath)

    if inspect.isclass(resolved):
        return _resolve_interface(resolved)
    else:
        raise ValueError(f"unable to resolve class {classpath}")


def _resolve_interface(cls):
    assert inspect.isclass(cls)

    from ..interface import Interface
    from ..state import State

    if issubclass(cls, Interface) and not issubclass(cls, State):
        return type(f"{cls.__name__}State", (State, cls), {})
    else:
        return cls


def _get_rtype(fn, default_rtype):
    if is_op(fn):
        return fn.rtype

    rtype = default_rtype

    if inspect.isfunction(fn):
        annotation = inspect.signature(fn).return_annotation
        rtype = resolve_class(fn, annotation, rtype)

    return rtype
