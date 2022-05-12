import inspect
import logging

from ..uri import uri, URI
from ..context import to_json


class Meta(type):
    """The metaclass of a :class:`Model` which provides support for `form_of` and `to_json`."""

    def parents(cls):
        from ..state import State

        parents = []
        for parent in cls.mro()[1:]:
            if issubclass(parent, State):
                if uri(parent) != uri(cls):
                    parents.append(parent)

        return parents

    def __form__(cls):
        from ..app import Model  # TODO: remove this dependency from Meta
        from ..scalar.ref import form_of, get_ref
        from ..state import State

        parents = [c for c in cls.parents() if not issubclass(c, Model)]
        parent_members = dict(inspect.getmembers(parents[0](form=URI("self")))) if parents else {}

        instance = cls(form=URI("self"))
        instance_header = get_ref(instance, "self")

        form = {}
        for name, attr in inspect.getmembers(instance):
            if name.startswith('_') or isinstance(attr, URI):
                continue
            elif hasattr(attr, "hidden") and attr.hidden:
                continue
            elif name in parent_members:
                if attr is parent_members[name] or attr == parent_members[name]:
                    logging.debug(f"{attr} is identical to its parent, won't be defined explicitly in {cls}")
                    continue
                elif hasattr(attr, "__code__") and hasattr(parent_members[name], "__code__"):
                    if attr.__code__ is parent_members[name].__code__:
                        logging.debug(f"{attr} is identical to its parent, won't be defined explicitly in {cls}")
                        continue

                logging.info(f"{attr} ({name}) overrides a parent method and will be explicitly defined in {cls}")
            elif inspect.ismethod(attr) and inspect.isclass(attr.__self__):
                # it's a @classmethod
                continue

            if isinstance(attr, State):
                while isinstance(attr, State):
                    attr = form_of(attr)

                if isinstance(attr, URI):
                    continue

                form[name] = attr
            if isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(instance_header, name))
            else:
                form[name] = attr

        return form

    def __json__(cls):
        from ..state import Class, State
        from ..scalar.ref import form_of

        parents = cls.parents()

        if not parents or uri(parents[0]).startswith(uri(State)):
            return {str(uri(Class)): to_json(form_of(cls))}
        else:
            return {str(uri(parents[0])): to_json(form_of(cls))}


class MethodStub(object):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"cannot call the instance method {self.form} from a static context")

    def method(self, header, name):
        return self.dtype(header, self.form, name)
