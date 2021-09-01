"""Reference types."""

from tinychain.reflect import is_conditional
from tinychain.util import deanonymize, requires, to_json, uri, URI


class Ref(object):
    """A reference to a :class:`State`."""

    __uri__ = URI("/state/scalar/ref")


class After(Ref):
    """A flow control operator used to delay execution conditionally."""

    __uri__ = uri(Ref) + "/after"

    def __init__(self, when, then):
        self.when = when
        self.then = then

    def __deps__(self):
        deps = set()
        deps.update(requires(self.when))
        deps.update(requires(self.then))
        return deps

    def __json__(self):
        return {str(uri(self)): to_json([self.when, self.then])}

    def __ns__(self, cxt):
        deanonymize(self.when, cxt)
        deanonymize(self.then, cxt)

        if is_conditional(self.when):
            self.when = reference(cxt, self.when)


class Before(Ref):
    """A flow control operator used to delay execution conditionally."""

    __uri__ = uri(Ref) + "/before"

    def __init__(self, when, then):
        self.when = when
        self.then = then

    def __deps__(self):
        deps = set()
        deps.update(requires(self.when))
        deps.update(requires(self.then))
        return deps

    def __json__(self):
        return {str(uri(self)): to_json([self.when, self.then])}

    def __ns__(self, cxt):
        deanonymize(self.when, cxt)
        deanonymize(self.then, cxt)

        if is_conditional(self.when):
            self.when = reference(cxt, self.when)


class Case(Ref):
    """A flow control operator used to branch execution conditionally."""

    __uri__ = uri(Ref) + "/case"

    def __init__(self, cond, switch, case):
        self.cond = cond
        self.switch = switch
        self.case = case

    def __deps__(self):
        deps = set()
        deps.update(requires(self.cond))
        deps.update(requires(self.switch))
        deps.update(requires(self.case))
        return deps

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.switch, self.case])}

    def __ns__(self, cxt):
        deanonymize(self.cond, cxt)
        deanonymize(self.switch, cxt)
        deanonymize(self.case, cxt)


class If(Ref):
    """A flow control operator used to resolve a :class:`State` conditionally."""

    __uri__ = uri(Ref) + "/if"

    def __init__(self, cond, then, or_else=None):
        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __deps__(self):
        deps = set()
        deps.update(requires(self.cond))
        deps.update(requires(self.then))
        deps.update(requires(self.or_else))
        return deps

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.then, self.or_else])}

    def __ns__(self, cxt):
        deanonymize(self.cond, cxt)
        deanonymize(self.then, cxt)
        deanonymize(self.or_else, cxt)

        if is_conditional(self.cond):
            self.cond = reference(cxt, self.cond)


class While(Ref):
    """A flow control operator to execute a closure repeatedly until a condition is met."""

    __uri__ = uri(Ref) + "/while"

    def __init__(self, cond, op, state):
        self.cond = cond
        self.op = op
        self.state = state

    def __deps__(self):
        deps = set()
        deps.update(requires(self.cond))
        deps.update(requires(self.op))
        deps.update(requires(self.state))
        return deps

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.op, self.state])}

    def __ns__(self, cxt):
        deanonymize(self.cond, cxt)
        deanonymize(self.op, cxt)
        deanonymize(self.state, cxt)


class With(Ref):
    """A flow control operator to limit the scope of a lambda Op"""

    __uri__ = uri(Ref) + "/with"

    def __init__(self, capture, op):
        self.capture = []

        for ref in capture:
            captured = uri(ref).subject()

            if captured is None:
                raise ValueError(f"With can only capture states with an ID in the current context, not {ref}")

            self.capture.append(captured)

        self.op = op

    def __deps__(self):
        return set(self.capture)

    def __json__(self):
        return {str(uri(self)): to_json([self.capture, self.op])}

    def __ns__(self, cxt):
        deanonymize(self.capture, cxt)
        deanonymize(self.op, cxt)


class Op(Ref):
    """A reference to an :class:`Op`."""

    __uri__ = uri(Ref) + "/op"

    def __init__(self, subject, args):
        self.subject = subject
        self.args = args

    def __deps__(self):
        deps = set()
        deps.update(requires(self.subject))
        deps.update(requires(self.args))
        return deps

    def __json__(self):
        if isinstance(self.subject, Ref):
            subject = self.subject
        else:
            subject = uri(self.subject)

        return {str(subject): to_json(self.args)}

    def __ns__(self, cxt):
        deanonymize(self.subject, cxt)
        deanonymize(self.args, cxt)


class Get(Op):
    """A reference to an instance of :class:`Op.Get`."""

    __uri__ = uri(Op) + "/get"

    def __init__(self, subject, key=None):
        if subject is None:
            raise ValueError("Get op ref subject cannot be None")

        Op.__init__(self, subject, (key,))

    def __json__(self):
        if isinstance(self.subject, Ref):
            subject = self.subject
            is_scalar = True
        else:
            subject = uri(self.subject)
            if subject is None:
                raise ValueError(f"subject of Get op ref {self.subject} ({type(self.subject)}) has no URI")

            is_scalar = subject.startswith("/state/scalar")

        if is_scalar:
            (value,) = self.args
            return {str(subject): to_json(value)}
        else:
            return {str(subject): to_json(self.args)}

    def __repr__(self):
        return f"GET Op ref {self.subject} {self.args}"


class Put(Op):
    """A reference to an instance of :class:`Op.Put`."""

    __uri__ = uri(Op) + "/put"

    def __init__(self, subject, key, value):
        Op.__init__(self, subject, (key, value))

    def __repr__(self):
        return f"PUT Op ref {self.subject} {self.args}"


class Post(Op):
    """A reference to an instance of :class:`Op.Post`."""

    __uri__ = uri(Op) + "/post"

    def __init__(self, subject, args):
        Op.__init__(self, subject, args)

    def __repr__(self):
        return f"POST Op ref {self.subject} {self.args}"


class Delete(Op):
    """A reference to an instance of :class:`Op.Delete`."""

    __uri__ = uri(Op) + "/delete"

    def __init__(self, subject, key=None):
        Op.__init__(self, subject, key)

    def __json__(self):
        return {str(uri(self)): to_json([self.subject, self.args])}

    def __repr__(self):
        return f"DELETE Op ref {self.subject} {self.args}"


class MethodSubject(object):
    def __init__(self, subject, method_name):
        if uri(subject).startswith('$'):
            self.__uri__ = uri(subject).append(method_name)
        else:
            self.__uri__ = None

        self.subject = subject
        self.method_name = method_name

    def __deps__(self):
        return requires(self.subject)

    def __ns__(self, cxt):
        if uri(self):
            return

        name = f"{self.subject.__class__.__name__}_{format(id(self.subject), 'x')}"
        self.__uri__ = URI(name).append(self.method_name)

        if name not in cxt.form:
            setattr(cxt, name, self.subject)

    def __json__(self):
        if self.__uri__ is None:
            raise ValueError(f"cannot call method {self.method_name} on an anonymous subject {self.subject}")

        return to_json(uri(self))

    def __str__(self):
        if self.__uri__ is None:
            return str(uri(self.subject).append(self.method_name))
        else:
            return str(uri(self))

    def assign_uri(self, subject_uri):
        assert self.__uri__ is None
        self.__uri__ = subject_uri.append(self.method_name)


def reference(cxt, state):
    name = f"{state.__class__.__name__}_{id(state)}"
    setattr(cxt, name, state)
    return URI(name)
