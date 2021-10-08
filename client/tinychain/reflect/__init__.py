import inspect

from pydoc import locate

from tinychain.state import Class, Instance, Map, State
from tinychain.util import deanonymize, form_of, to_json, uri, URI

from .meta import gen_headers, Meta


class Object(Class, metaclass=Meta):
    def __init__(self, form=None):
        self.class_uri = uri(self.__class__)
        super().__init__(form)

    def __json__(self):
        form = form_of(self)
        if is_ref(form):
            return to_json(form)
        else:
            return {str(self.class_uri): to_json(form)}

    def __ns__(self, cxt):
        deanonymize(super(), cxt)

        if uri(self.__class__) == uri(Class):
            name = f"Class_{self.__class__.__name__}_{format(id(self.__class__), 'x')}"

            if not cxt.is_defined(name):
                setattr(cxt, name, self.__class__)

            self.class_uri = URI(name)


def _get_rtype(fn, default_rtype):
    if is_op(fn):
        return fn.rtype

    rtype = default_rtype

    if inspect.isfunction(fn):
        annotation = inspect.signature(fn).return_annotation
        rtype = resolve_class(fn, annotation, rtype)

    return rtype


def is_conditional(state):
    from tinychain.ref import Case, If
    from tinychain.state import State

    if isinstance(state, State):
        return is_conditional(form_of(state))

    return isinstance(state, Case) or isinstance(state, If)


def is_none(state):
    from tinychain.value import Nil

    return state is None or state == Nil


def is_op(fn):
    from .method import Method
    from .op import Op

    return isinstance(fn, Method) or isinstance(fn, Op)


def is_ref(state):
    from tinychain.ref import MethodSubject, Ref

    if isinstance(state, Ref) or isinstance(state, URI) or isinstance(state, MethodSubject):
        return True
    elif hasattr(state, "__form__"):
        return is_ref(form_of(state))
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
