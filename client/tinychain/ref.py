"""Reference types."""

from .util import form_of, uri, URI, to_json

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


class Case(Ref):
    """A flow control operator used to branch execution conditionally."""

    __uri__  = uri(Ref) + "/case"

    def __init__(self, cond, switch, case):
        self.cond = cond
        self.switch = switch
        self.case = case

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.switch, self.case])}


class If(Ref):
    """A flow control operator used to resolve a :class:`State` conditionally."""

    __uri__ = uri(Ref) + "/if"

    def __init__(self, cond, then, or_else):
        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.then, self.or_else])}


class OpRef(Ref):
    """A reference to an :class:`Op`."""

    __uri__ = uri(Ref) + "/op"

    def __init__(self, subject, args):
        if isinstance(subject, Ref):
            self.subject = subject
        else:
            self.subject = uri(subject)

        self.args = args

    def __json__(self):
        return {str(self.subject): to_json(self.args)}


class GetOpRef(OpRef):
    """A reference to an instance of :class:`Op.Get`."""

    __uri__ = uri(OpRef) + "/get"

    def __init__(self, subject, key=None):
        if str(uri(subject)).startswith("/state/scalar"):
            OpRef.__init__(self, subject, key)
        else:
            OpRef.__init__(self, subject, (key,))


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

