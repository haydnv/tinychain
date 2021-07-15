import inspect

from tinychain.state import State
from tinychain.util import form_of, to_json, uri, URI


def gen_headers(instance):
    for name, attr in inspect.getmembers(instance):
        if name.startswith('_'):
            continue

        if isinstance(attr, MethodStub):
            setattr(instance, name, attr.method(instance, name))


class Meta(type):
    """The metaclass of a :class:`State` which provides support for `form_of` and `to_json`."""

    def __form__(cls):
        mro = cls.mro()
        if len(mro) < 2:
            raise ValueError("Tinychain class must extend a subclass of State")

        parent_members = dict(inspect.getmembers(mro[1](URI("self"))))

        class Header(cls):
            pass

        instance_uri = URI("self")
        header = Header(instance_uri)
        instance = cls(instance_uri)

        for name, attr in inspect.getmembers(instance):
            if name.startswith('_'):
                continue

            if isinstance(attr, MethodStub):
                setattr(header, name, attr.method(instance, name))
            elif isinstance(attr, State):
                setattr(header, name, type(attr)(instance_uri.append(name)))
            else:
                setattr(header, name, attr)

        form = {}
        for name, attr in inspect.getmembers(instance):
            if name.startswith('_'):
                continue
            elif name in parent_members:
                if attr is parent_members[name]:
                    continue
                elif hasattr(attr, "__code__") and hasattr(parent_members[name], "__code__"):
                    if attr.__code__ is parent_members[name].__code__:
                        continue

            if isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(header, name))
            else:
                form[name] = attr

        return form

    def __json__(cls):
        mro = cls.mro()
        if len(mro) < 2:
            raise ValueError("Tinychain class must extend a subclass of State")

        parent = mro[1]
        return {str(uri(parent)): to_json(form_of(cls))}


class MethodStub(object):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "cannot call a MethodStub; use tc.use(<class>) for callable method references")

    def method(self, header, name):
        return self.dtype(header, self.form, name)
