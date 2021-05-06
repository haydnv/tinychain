"""Reference types."""

from .util import *

class Ref(object):
    """A reference to a :class:`State`."""

    __uri__ = URI("/state/scalar/ref")


class After(Ref):
    """A flow control operator used to delay execution conditionally."""

    __uri__ = uri(Ref) + "/after"

    def __init__(self, when, then):
        self.when = when
        self.then = then

    def __json__(self):
        return {str(uri(self)): to_json([self.when, self.then])}

    def __ns__(self, cxt):
        deanonymize(self.when, cxt)
        deanonymize(self.then, cxt)


class Case(Ref):
    """A flow control operator used to branch execution conditionally."""

    __uri__  = uri(Ref) + "/case"

    def __init__(self, cond, switch, case):
        self.cond = cond
        self.switch = switch
        self.case = case

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.switch, self.case])}

    def __ns__(self, cxt):
        deanonymize(self.cond, cxt)
        deanonymize(self.switch, cxt)
        deanonymize(self.then, cxt)


class If(Ref):
    """A flow control operator used to resolve a :class:`State` conditionally."""

    __uri__ = uri(Ref) + "/if"

    def __init__(self, cond, then, or_else):
        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.then, self.or_else])}

    def __ns__(self, cxt):
        deanonymize(self.cond, cxt)
        deanonymize(self.then, cxt)
        deanonymize(self.or_else, cxt)


class OpRef(Ref):
    """A reference to an :class:`Op`."""

    __uri__ = uri(Ref) + "/op"

    def __init__(self, subject, args):
        self.subject = subject
        self.args = args

    def __json__(self):
        if isinstance(self.subject, Ref):
            subject = self.subject
        else:
            subject = uri(self.subject)

        return {str(subject): to_json(self.args)}

    def __ns__(self, cxt):
        if isinstance(self.subject, MethodSubject):
            deanonymize(self.subject, cxt)
            deanonymize(self.args, cxt)


class GetOpRef(OpRef):
    """A reference to an instance of :class:`Op.Get`."""

    __uri__ = uri(OpRef) + "/get"

    def __init__(self, subject, key=None):
        OpRef.__init__(self, subject, (key,))

    def __json__(self):
        if isinstance(self.subject, Ref):
            subject = self.subject
        else:
            subject = uri(self.subject)

        if str(uri(subject)).startswith("/state/scalar"):
            (value,) = self.args
            return {str(subject): to_json(value)}
        else:
            return {str(subject): to_json(self.args)}


class PutOpRef(OpRef):
    """A reference to an instance of :class:`Op.Put`."""

    __uri__ = uri(OpRef) + "/put"

    def __init__(self, subject, key, value):
        OpRef.__init__(self, subject, (key, value))


class PostOpRef(OpRef):
    """A reference to an instance of :class:`Op.Post`."""

    __uri__ = uri(OpRef) + "/post"

    def __init__(self, subject, **kwargs):
        OpRef.__init__(self, subject, kwargs)


class DeleteOpRef(OpRef):
    """A reference to an instance of :class:`Op.Delete`."""

    __uri__ = uri(OpRef) + "/delete"

    def __init__(self, subject, key=None):
        OpRef.__init__(self, subject, key)

    def __json__(self):
        return {str(uri(self)): to_json([self.subject, self.args])}


OpRef.Get = GetOpRef
OpRef.Put = PutOpRef
OpRef.Post = PostOpRef
OpRef.Delete = DeleteOpRef


class MethodSubject(object):
    def __init__(self, subject, method_name):
        self.__uri__ = None
        self.subject = subject
        self.method_name = method_name

    def __ns__(self, cxt):
        name = cxt.generate_name(self.subject.__class__.__name__)
        self.__uri__ = URI(name).append(self.method_name)
        setattr(cxt, name, self.subject)

    def __json__(self):
        if self.__uri__ is None:
            raise ValueError(
                f"cannot call method {self.method_name} on an anonymous subject {self.subject}")

        return to_json(uri(self))

    def __str__(self):
        if self.__uri__ is None:
            return str(uri(self.subject).append(self.method_name))
        else:
            return str(uri(self))

