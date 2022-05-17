"""Reference types"""

import logging

from ...uri import uri, URI
from ...context import deanonymize, to_json


class Ref(object):
    """A reference to a :class:`State`. Prefer to construct a subclass like :class:`If` or :class:`Get`."""

    __uri__ = URI("/state/scalar/ref")

    def __same__(self, other):
        from .helpers import same_as
        return same_as(self.__args__(), other.__args__())


class FlowControl(Ref):
    """A flow control like :class:`If` or :class:`After`."""


class After(FlowControl):
    """
    A flow control operator used to delay execution conditionally.

    Args:
        when (State or Ref): A reference to resolve before resolving `then`.

        then (State or Ref): The state to return when this `Ref` is resolved.

    Example:
            num_rows = After(table.insert(["key"], ["value"]), table.count())`
    """

    __uri__ = uri(Ref) + "/after"

    def __init__(self, when, then):
        from .helpers import is_conditional

        if is_conditional(then):
            raise ValueError(f"After does not support a conditional clause: {then}")

        self.when = when
        self.then = then

    def __args__(self):
        return [self.when, self.then]

    def __json__(self):
        return {str(uri(self)): to_json([self.when, self.then])}

    def __ns__(self, cxt, name_hint):
        from .helpers import is_conditional, reference

        deanonymize(self.when, cxt, name_hint + "_when")
        deanonymize(self.then, cxt, name_hint + "_then")

        if is_conditional(self.when):
            self.when = reference(cxt, self.when, name_hint + "_when")

    def __repr__(self):
        return f"After({self.when}, {self.then})"


class Case(FlowControl):
    """
    A flow control used to branch execution conditionally.

    Args:
        cond (Value or Ref): The Value to match each `switch` case against.

        switch (Tuple): A Tuple of values to match `cond` against.

        case: A Tuple of possible States to resolve, with length = `switch.len() + 1`, where the last is the default.

    Example:
        `equals_one = Case(1, [0, 1], [False, True, False])`

    Raises:
        `BadRequestError` if `case` is the wrong length or if `cond` or `switch` contains a nested conditional
    """

    __uri__ = uri(Ref) + "/case"

    def __init__(self, cond, switch, case):
        self.cond = cond
        self.switch = switch
        self.case = case

    def __args__(self):
        return [self.cond, self.switch, self.case]

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.switch, self.case])}

    def __ns__(self, cxt, name_hint):
        deanonymize(self.cond, cxt, name_hint + "_cond")
        deanonymize(self.switch, cxt, name_hint + "_switch")
        deanonymize(self.case, cxt, name_hint + "_case")

    def __repr__(self):
        return f"Cast({self.cond}, {self.switch}, {self.case})"


class If(FlowControl):
    """
    A flow control used to branch execution conditionally.

    Args:
        cond (Bool or Ref): The condition determines which branch to execute

        then (State or Ref): The State to resolve if `cond` resolves to `True`

        or_else (State or Ref): The State to resolve if `cond` resolves to `False`

    Raises:
        `BadRequestError` if `cond` does not resolve to a :class:`Bool` or in case of a nested conditional
    """

    __uri__ = uri(Ref) + "/if"

    def __init__(self, cond, then, or_else=None):
        from .helpers import is_conditional

        if is_conditional(cond):
            raise ValueError(f"If does not support nested conditionals: {cond}")

        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __args__(self):
        return [self.cond, self.then, self.or_else]

    def __json__(self):
        from .helpers import form_of

        # TODO: move this short-circuit condition into a helper function called `cond` that returns a typed `If`
        if isinstance(form_of(self.cond), bool):
            if form_of(self.cond):
                return to_json(self.then)
            else:
                return to_json(self.or_else)

        return {str(uri(self)): to_json([self.cond, self.then, self.or_else])}

    def __ns__(self, cxt, name_hint):
        from .helpers import is_conditional, is_op_ref, reference

        deanonymize(self.cond, cxt, name_hint + "_cond")
        deanonymize(self.then, cxt, name_hint + "_then")
        deanonymize(self.or_else, cxt, name_hint + "_or_else")

        if is_conditional(self.cond) or is_op_ref(self.cond):
            self.cond = reference(cxt, self.cond, name_hint + "_cond")

    def __repr__(self):
        from .helpers import form_of

        # TODO: move this short-circuit condition into a helper function called `cond` that returns a typed `If`
        if isinstance(form_of(self.cond), bool):
            if self.cond:
                return str(self.then)
            else:
                return str(self.or_else)

        if self.or_else:
            return f"If({self.cond}, {self.then}, {self.or_else})"
        else:
            return f"If({self.cond}, {self.then})"


class While(FlowControl):
    """
    A flow control operator to execute a closure repeatedly until a condition is met.

    Args:
        cond (Post): The condition which determines whether to terminate the loop.

        step (Post): One cycle of this `While` loop.

        state (State): The initial state of the loop.
    """

    __uri__ = uri(Ref) + "/while"

    def __init__(self, cond, op, state=None):
        self.cond = cond
        self.op = op
        self.state = state

    def __args__(self):
        return [self.cond, self.op, self.state]

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.op, self.state])}

    def __ns__(self, cxt, name_hint):
        deanonymize(self.cond, cxt, name_hint + "_cond")
        deanonymize(self.op, cxt, name_hint + "_op")
        deanonymize(self.state, cxt, name_hint + "_state")

    def __repr__(self):
        return f"While({self.cond}, {self.op}, {self.state})"


class With(FlowControl):
    """
    Capture state from an enclosing context. Prefer using the `closure` decorator to construct a `With` automatically.

    Args:
        capture (Iterable): A Python iterable with the `Id` of each `State` to capture from the outer `Op` context.

        op (Op): The Op to close over.
    """

    __uri__ = uri(Ref) + "/with"

    def __init__(self, capture, op):
        self.capture = []

        for ref in capture:
            id = uri(ref).id()
            if id is None:
                raise ValueError(f"With can only capture states with an ID in the current context, not {ref}")

            self.capture.append(URI(id))

        self.op = op

        if hasattr(op, "rtype"):
            self.rtype = op.rtype

    def __args__(self):
        return [self.capture, self.op]

    def __json__(self):
        return {str(uri(self)): to_json([self.capture, self.op])}

    def __ns__(self, _cxt, _name_hint):
        pass

    def __ref__(self, name):
        from .helpers import get_ref
        return get_ref(self.op, name)

    def __repr__(self):
        return f"With({self.capture}, {self.op})"


class Op(Ref):
    """A resolvable reference to an :class:`Op`."""

    __uri__ = uri(Ref) + "/op"

    def __init__(self, subject, args, debug_name=None):
        self._debug_name = debug_name
        self.subject = subject
        self.args = args

    def __repr__(self):
        return self._debug_name if self._debug_name else f"{self.__class__.__name__} {self.subject}, {self.args}"

    def __args__(self):
        from .helpers import is_op_ref

        subject = [self.subject] if is_op_ref(self.subject, allow_literals=False) else []
        return subject + [arg for arg in list(self.args) if is_op_ref(arg)]

    def __json__(self):
        from .helpers import form_of

        if hasattr(self.subject, "__form__"):
            subject = form_of(self.subject)
        else:
            subject = self.subject

        if uri(subject) is None:
            raise ValueError(f"subject {self.subject} of {self} has no URI")

        return {str(uri(subject)): to_json(self.args)}

    def __ns__(self, cxt, name_hint):
        deanonymize(self.subject, cxt, name_hint + "_subject")

    def __same__(self, other):
        from .helpers import same_as
        return same_as(self.subject, other.subject) and same_as(self.args, other.args)


class Get(Op):
    """
    A `Get` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        key (Value or Ref): The `key` with which to call this `Op`.
    """

    __uri__ = uri(Op) + "/get"

    def __init__(self, subject, key=None, debug_name=None):
        if subject is None:
            raise ValueError("Get op ref subject cannot be None")

        Op.__init__(self, subject, (key,), debug_name)

    def __json__(self):
        from .helpers import is_ref

        if isinstance(self.subject, Ref):
            subject = uri(self.subject)
            is_scalar = False
        elif isinstance(self.subject, URI) or hasattr(self.subject, "__uri__"):
            subject = uri(self.subject)
            if subject is None:
                raise ValueError(f"subject of Get op ref {self.subject} ({type(self.subject)}) has no URI")

            is_scalar = subject.startswith("/state/scalar")
        else:
            raise ValueError(f"subject of {self} has no URI")

        if is_scalar and not is_ref(self.args):
            (value,) = self.args
            return {str(subject): to_json(value)}
        else:
            return {str(subject): to_json(self.args)}

    def __repr__(self):
        if self._debug_name:
            return str(self._debug_name)
        else:
            return f"GET {repr(self.subject)}: {repr(self.args)}"

    def __ns__(self, cxt, name_hint):
        from .helpers import is_op_ref, reference

        super().__ns__(cxt, name_hint)

        deanonymize(self.args, cxt, name_hint + "_key")

        if is_op_ref(self.args):
            (key,) = self.args
            _log_anonymous(key)
            key = reference(cxt, key, name_hint + "_key")
            self.args = (key,)


class Put(Op):
    """
    A `Put` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        key (Value or Ref): The `key` with which to call this `Op`.

        value (State or Ref): The `value` with which to call this `Op`.
    """

    __uri__ = uri(Op) + "/put"

    def __init__(self, subject, key, value):
        Op.__init__(self, subject, (key, value))

    def __repr__(self):
        key, value = self.args
        return f"PUT {repr(self.subject)}: {repr(key)} <- {repr(value)}"

    def __ns__(self, cxt, name_hint):
        from .helpers import is_op_ref, reference

        super().__ns__(cxt, name_hint)

        key, value = self.args
        deanonymize(key, cxt, name_hint + "_key")
        deanonymize(value, cxt, name_hint + "_value")

        if is_op_ref(key):
            _log_anonymous(key)
            key = reference(cxt, key, name_hint + "key")

        if is_op_ref(value):
            _log_anonymous(value)
            value = reference(cxt, value, name_hint + "_value")

        self.args = (key, value)


class Post(Op):
    """
    A `Post` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        args (Map or Ref): The parameters with which to call this `Op`.
    """

    __uri__ = uri(Op) + "/post"

    def __init__(self, subject, args, debug_name=None):
        if not hasattr(args, "__iter__"):
            raise ValueError("POST Op ref requires named parameters (try using a Python dict)")

        Op.__init__(self, subject, args, debug_name)

    def __args__(self):
        from .helpers import is_op_ref

        args = [self.subject] if is_op_ref(self.subject) else []
        return args + ([self.args.values()] if is_op_ref(self.args) else [])

    def __repr__(self):
        if self._debug_name:
            return str(self._debug_name)
        else:
            return f"POST {repr(self.subject)}: {self.args}"

    def __ns__(self, cxt, name_hint):
        from .helpers import is_op_ref, reference

        super().__ns__(cxt, name_hint)

        if not isinstance(self.args, dict):
            raise ValueError(f"POST arguments must be a Python dict, not {self.args}")

        args = {}
        for name, arg in self.args.items():
            deanonymize(arg, cxt, name_hint + f"_{name}")

            if is_op_ref(arg):
                _log_anonymous(arg)
                args[name] = reference(cxt, arg, name_hint + f"_{name}")
            else:
                args[name] = arg

        self.args = args


class Delete(Op):
    """
    A `Delete` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        key (Value or Ref): The `key` with which to call this `Op`.
    """

    __uri__ = uri(Op) + "/delete"

    def __init__(self, subject, key=None):
        Op.__init__(self, subject, key)

    def __json__(self):
        return {str(uri(self)): to_json([self.subject, self.args])}

    def __repr__(self):
        return f"DELETE {repr(self.subject)}: {repr(self.args)}"

    def __ns__(self, cxt, name_hint):
        from .helpers import is_op_ref, reference

        super().__ns__(cxt, name_hint)
        deanonymize(self.args, cxt, name_hint + "_key")

        if is_op_ref(self.args):
            _log_anonymous(self.args)
            self.args = reference(cxt, self.args, name_hint + "_key")


class MethodSubject(object):
    def __init__(self, subject, method_name=""):
        if uri(subject).startswith('$'):
            self.__uri__ = uri(subject).append(method_name)

        self.subject = subject
        self.method_name = method_name

    def __args__(self):
        from .helpers import is_op_ref
        return [self.subject] if is_op_ref(self.subject) else []

    def __ns__(self, cxt, name_hint):
        from .helpers import same_as

        i = 0
        name = name_hint
        while name in cxt and not same_as(getattr(cxt, name), self.subject):
            name = f"{name_hint}_{i}" if i > 0 else name_hint
            i += 1

        auto_uri = URI(name).append(self.method_name)

        if hasattr(self, "__uri__"):
            if uri(self) == auto_uri and name not in cxt:
                logging.debug(f"auto-assigning name {name} to {self.subject} in {cxt}")
                setattr(cxt, name, self.subject)
            else:
                return

        elif uri(self.subject).startswith("/state"):
            self.__uri__ = auto_uri

            if name not in cxt:
                logging.debug(f"auto-assigning name {name} to {self.subject} in {cxt}")
                setattr(cxt, name, self.subject)

            deanonymize(self.subject, cxt, name_hint)

        else:
            deanonymize(self.subject, cxt, name_hint)
            self.__uri__ = uri(self.subject).append(self.method_name)

    def __json__(self):
        if self.__uri__ is None:
            raise ValueError(f"cannot call method {self.method_name} on an anonymous subject {self.subject}")

        return to_json(uri(self))

    def __str__(self):
        if not hasattr(self, "__uri__"):
            return str(uri(self.subject).append(self.method_name))
        else:
            return str(uri(self))

    def __repr__(self):
        return f"{repr(self.subject)}/{self.method_name}"


def _log_anonymous(arg):
    from .helpers import is_op_ref, is_write_op_ref

    if is_write_op_ref(arg):
        logging.warning(f"assigning auto-generated name to the result of {arg}")
    elif is_op_ref(arg):
        logging.info(f"assigning auto-generated name to the result of {arg}")
