import inspect
import typing

from pydoc import locate


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
                raise TypeError(f"missing required positional argument {name} (arguments are *{args})")
            else:
                params[name] = param.default
        elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if i < len(args):
                params[name] = args[i]
                i += 1
            elif name in kwargs:
                params[name] = kwargs[name]
            elif param.default is inspect.Parameter.empty:
                raise TypeError(f"missing required argument {name} (arguments are *{args}, **{kwargs})")
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
        return Tuple[annotation]
    elif typing.get_origin(annotation) is dict:
        from ..generic import Map
        return Map[annotation]
    elif inspect.isclass(annotation):
        from ..generic import resolve_interface
        return resolve_interface(annotation)
    elif callable(annotation):
        # assume that this is a generic alias which will construct an instance when called
        return annotation

    classpath = f"{subject.__module__}.{annotation}"
    resolved = locate(classpath)

    if inspect.isclass(resolved):
        from ..generic import resolve_interface
        return resolve_interface(resolved)
    else:
        raise ValueError(f"unable to resolve class {classpath}")


def get_rtype(fn, default_rtype):
    from ..scalar.ref import is_op

    if is_op(fn):
        return fn.rtype

    rtype = default_rtype

    if inspect.isfunction(fn):
        annotation = inspect.signature(fn).return_annotation
        rtype = resolve_class(fn, annotation, rtype)

    return rtype
