import inspect
import logging

from tinychain.state import State
from tinychain.util import form_of, get_ref, to_json, uri, URI


class MethodStub(object):
    def __init__(self, dtype, form):
        self.dtype = dtype
        self.form = form

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"cannot call a MethodStub; use tc.use(<class>) for callable method references")

    def method(self, header, name):
        return self.dtype(header, self.form, name)


def header(cls):
    instance_uri = URI("self")

    class Header(cls):
        pass

    try:
        header = Header(form=instance_uri)
        instance = cls(form=instance_uri)
    except Exception as e:
        raise RuntimeError(f"unable to generate headers for {cls}", e)

    for name, attr in inspect.getmembers(instance):
        if name.startswith('_') or isinstance(attr, URI):
            continue
        elif inspect.ismethod(attr) and attr.__self__ is cls:
            # it's a @classmethod
            continue

        if isinstance(attr, MethodStub):
            setattr(header, name, attr.method(instance, name))
        elif isinstance(attr, State):
            member_uri = instance_uri.append(name)
            attr_ref = get_ref(attr, member_uri)

            if not uri(attr_ref) == member_uri:
                raise RuntimeError(f"failed to assign URI {member_uri} to instance attribute {attr_ref} "
                                   + f"(assigned URI is {uri(attr_ref)})")

            setattr(header, name, attr_ref)
        else:
            setattr(header, name, attr)

    return instance, header


class Meta(type):
    """The metaclass of a :class:`State` which provides support for `form_of` and `to_json`."""

    def __form__(cls):
        mro = [c for c in cls.mro()[1:] if issubclass(c, State)]
        if not mro:
            raise ValueError("TinyChain class must extend a subclass of State")

        parent_members = dict(inspect.getmembers(mro[0](form=URI("self"))))

        instance, instance_header = header(cls)

        form = {}
        for name, attr in inspect.getmembers(instance):
            if name.startswith('_') or isinstance(attr, URI):
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
            elif inspect.ismethod(attr) and attr.__self__ is cls:
                # it's a @classmethod
                continue

            if isinstance(attr, MethodStub):
                form[name] = to_json(attr.method(instance_header, name))
            else:
                form[name] = attr

        return form

    def __json__(cls):
        mro = cls.mro()
        if len(mro) < 2:
            raise ValueError("TinyChain class must extend a subclass of State")

        parent = mro[1]
        return {str(uri(parent)): to_json(form_of(cls))}
