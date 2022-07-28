"""Reference types"""
import typing

from ...json import to_json
from ...uri import URI


class Ref(object):
    """A reference to a :class:`State`. Prefer to construct a subclass like :class:`If` or :class:`Get`."""

    __uri__ = URI("/state/scalar/ref")

    def __same__(self, other):
        from .functions import same_as
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

    __uri__ = URI(Ref) + "/after"

    def __init__(self, when, then):
        from .functions import is_conditional

        if is_conditional(then):
            raise ValueError(f"After does not support a conditional clause: {then}")

        self.when = when
        self.then = then

    def __args__(self):
        return self.when, self.then

    def __json__(self):
        return {str(URI(self)): to_json([self.when, self.then])}

    def __ns__(self, cxt, name_hint):
        from .functions import is_conditional

        cxt.deanonymize(self.when, name_hint + "_when")
        cxt.deanonymize(self.then, name_hint + "_then")

        if is_conditional(self.when):
            cxt.assign(self.when, name_hint + "_when")

    def __repr__(self):
        return f"After({self.when}, {self.then})"


def after(when, then):
    """Delay execution of `then` until `when` is resolved."""

    from ...generic import autobox
    from ...state import State

    then = autobox(then)
    rtype = type(then) if isinstance(then, State) else State
    return rtype(form=After(when, then))


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

    __uri__ = URI(Ref) + "/case"

    def __init__(self, cond, switch, case):
        self.cond = cond
        self.switch = switch
        self.case = case

    def __args__(self):
        return self.cond, self.switch, self.case

    def __json__(self):
        return {str(URI(self)): to_json([self.cond, self.switch, self.case])}

    def __ns__(self, cxt, name_hint):
        cxt.deanonymize(self.cond, name_hint + "_cond")
        cxt.deanonymize(self.switch, name_hint + "_switch")
        cxt.deanonymize(self.case, name_hint + "_case")

    def __repr__(self):
        return f"Cast({self.cond}, {self.switch}, {self.case})"


def switch_case(cond, switch, case):
    """
    Resolve the `case` at the first index of `switch` which resolves to `True`,
    or the last `case` if no `switch` is `True`.
    """

    from ...generic import autobox, gcs
    from ...state import State

    case = autobox(case)
    if hasattr(case, "__iter__"):
        rtype = gcs(*[type(c) for c in case])
        rtype = rtype if issubclass(rtype, State) else State
    else:
        rtype = State

    return rtype(form=Case(cond, switch, case))


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

    __uri__ = URI(Ref) + "/if"

    def __init__(self, cond, then, or_else=None):
        from .functions import is_conditional

        if is_conditional(cond):
            raise ValueError(f"If does not support nested conditionals: {cond}")

        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __args__(self):
        return self.cond, self.then, self.or_else

    def __json__(self):
        from .functions import form_of

        # TODO: move this short-circuit condition into a helper function called `cond` that returns a typed `If`
        if isinstance(form_of(self.cond), bool):
            if form_of(self.cond):
                return to_json(self.then)
            else:
                return to_json(self.or_else)

        return {str(URI(self)): to_json([self.cond, self.then, self.or_else])}

    def __ns__(self, cxt, name_hint):
        from .functions import is_conditional, is_op_ref

        cxt.deanonymize(self.cond, name_hint + "_cond")
        cxt.deanonymize(self.then, name_hint + "_then")
        cxt.deanonymize(self.or_else, name_hint + "_or_else")

        if is_conditional(self.cond) or is_op_ref(self.cond):
            cxt.assign(self.cond, name_hint + "_cond")

    def __repr__(self):
        from .functions import form_of

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


def cond(cond, then, or_else=None):
    """Resolve either `then` or `or_else` conditionally based on the resolved value of `cond`."""

    from ...error import TinyChainError
    from ...generic import autobox, gcs
    from ...state import State

    then = autobox(then)
    or_else = autobox(or_else)

    if or_else is None:
        rtype = type(then) if isinstance(then, State) else State
    elif isinstance(then, TinyChainError):
        rtype = type(or_else) if isinstance(or_else, State) else State
    elif isinstance(or_else, TinyChainError):
        rtype = type(then) if isinstance(then, State) else State
    elif isinstance(then, State) and isinstance(or_else, State):
        rtype = gcs(type(then), type(or_else))
    else:
        rtype = State

    return rtype(form=If(cond, then, or_else))


class While(FlowControl):
    """
    A flow control operator to execute a closure repeatedly until a condition is met.

    Args:
        cond (Post): The condition which determines whether to terminate the loop.

        step (Post): One cycle of this `While` loop.

        state (State): The initial state of the loop.
    """

    __uri__ = URI(Ref) + "/while"

    def __init__(self, cond, op, state=None):
        self.cond = cond
        self.op = op
        self.state = state

    def __args__(self):
        return self.cond, self.op, self.state

    def __json__(self):
        return {str(URI(self)): to_json([self.cond, self.op, self.state])}

    def __ns__(self, cxt, name_hint):
        cxt.deanonymize(self.cond, name_hint + "_cond")
        cxt.deanonymize(self.op, name_hint + "_op")
        cxt.deanonymize(self.state, name_hint + "_state")

    def __repr__(self):
        return f"While({self.cond}, {self.op}, {self.state})"


def while_loop(cond, op, state: None):
    """Call `op` with `state` while `cond` is `True`."""

    from ...generic import autobox, Map
    from ...state import State

    state = autobox(state)
    rtype = type(state) if isinstance(state, Map) else Map[State]
    return rtype(form=While(cond, op, state))


class With(FlowControl):
    """
    Capture state from an enclosing context. Prefer using the `closure` decorator to construct a `With` automatically.

    Args:
        capture (Iterable): A Python iterable with the `Id` of each `State` to capture from the outer `Op` context.

        op (Op): The Op to close over.
    """

    __uri__ = URI(Ref) + "/with"

    def __init__(self, capture, op):
        self.capture = []

        for ref in capture:
            id = (ref if isinstance(ref, URI) else URI(ref)).id()
            if id is None:
                raise ValueError(f"With can only capture states with an ID in the current context, not {ref}")

            self.capture.append(URI(id))

        self.op = op

        if hasattr(op, "rtype"):
            self.rtype = op.rtype

    def __args__(self):
        return self.capture, self.op

    def __json__(self):
        return {str(URI(self)): to_json([self.capture, self.op])}

    def __ns__(self, _cxt, _name_hint):
        pass

    def __ref__(self, name):
        from .functions import get_ref
        return get_ref(self.op, name)

    def __repr__(self):
        return f"With({self.capture}, {self.op})"


class Op(Ref):
    """A resolvable reference to an :class:`Op`."""

    __uri__ = URI(Ref) + "/op"

    def __init__(self, subject, args, debug_name=None):
        self._debug_name = debug_name
        self.subject = subject
        self.args = args

    def __repr__(self):
        return self._debug_name if self._debug_name else f"{self.__class__.__name__} {self.subject}, {self.args}"

    def __args__(self):
        return self.subject, self.args, self._debug_name

    def __json__(self):
        from .functions import form_of

        if hasattr(self.subject, "__form__"):
            subject = form_of(self.subject)
        else:
            subject = self.subject

        if isinstance(subject, URI):
            pass
        elif not hasattr(subject, "__uri__"):
            raise ValueError(f"subject {self.subject} of {self} has no URI")
        elif URI(subject).startswith("/state") and URI(subject) == URI(type(subject)):
            raise RuntimeError(f"{self.subject} was not assigned a URI")

        subject = subject if isinstance(subject, URI) else URI(subject)
        return {str(subject): to_json(self.args)}

    def __ns__(self, cxt, name_hint):
        from .functions import is_literal, is_op_ref

        cxt.deanonymize(self.subject, name_hint + "_subject")

        if is_literal(self.subject) or is_op_ref(self.subject):
            cxt.assign(self.subject, name_hint + "_subject")

    def __same__(self, other):
        from .functions import same_as
        return same_as(self.subject, other.subject) and same_as(self.args, other.args)


class Get(Op):
    """
    A `Get` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        key (Value or Ref): The `key` with which to call this `Op`.
    """

    __uri__ = URI(Op) + "/get"

    def __init__(self, subject, key=None, debug_name=None):
        if subject is None:
            raise ValueError("Get op ref subject cannot be None")

        Op.__init__(self, subject, (key,), debug_name)

    def __args__(self):
        key, = self.args
        return self.subject, key, self._debug_name

    def __json__(self):
        from .functions import is_ref

        if isinstance(self.subject, Ref):
            subject = URI(self.subject)
            is_scalar = False
        elif isinstance(self.subject, URI) or hasattr(self.subject, "__uri__"):
            subject = self.subject if isinstance(self.subject, URI) else URI(self.subject)
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
        from .functions import is_op_ref

        super().__ns__(cxt, name_hint)

        cxt.deanonymize(self.args, name_hint + "_key")

        if is_op_ref(self.args):
            (key,) = self.args
            cxt.assign(key, name_hint + "_key")


class Put(Op):
    """
    A `Put` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        key (Value or Ref): The `key` with which to call this `Op`.

        value (State or Ref): The `value` with which to call this `Op`.
    """

    __uri__ = URI(Op) + "/put"

    def __init__(self, subject, key, value):
        Op.__init__(self, subject, (key, value))

    def __args__(self):
        (key, value) = self.args
        return self.subject, key, value

    def __repr__(self):
        key, value = self.args
        return f"PUT {repr(self.subject)}: {repr(key)} <- {repr(value)}"

    def __ns__(self, cxt, name_hint):
        from .functions import is_op_ref

        super().__ns__(cxt, name_hint)

        key, value = self.args
        cxt.deanonymize(key, name_hint + "_key")
        cxt.deanonymize(value, name_hint + "_value")

        if is_op_ref(key):
            cxt.assign(key, name_hint + "key")

        if is_op_ref(value):
            cxt.assign(value, name_hint + "_value")


class Post(Op):
    """
    A `Post` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        args (Map or Ref): The parameters with which to call this `Op`.
    """

    __uri__ = URI(Op) + "/post"

    def __init__(self, subject, args, debug_name=None):
        if not hasattr(args, "__iter__"):
            raise ValueError("POST Op ref requires named parameters (try using a Python dict)")

        Op.__init__(self, subject, args, debug_name)

    def __args__(self):
        return self.subject, self.args, self._debug_name

    def __repr__(self):
        if self._debug_name:
            return str(self._debug_name)
        else:
            return f"POST {repr(self.subject)}: {self.args}"

    def __ns__(self, cxt, name_hint):
        from .functions import is_op_ref

        super().__ns__(cxt, name_hint)

        if not isinstance(self.args, dict):
            raise ValueError(f"POST arguments must be a Python dict, not {self.args}")

        for name, arg in self.args.items():
            cxt.deanonymize(arg, name_hint + f"_{name}")

            if is_op_ref(arg):
                cxt.assign(arg, name_hint + f"_{name}")


class Delete(Op):
    """
    A `Delete` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        key (Value or Ref): The `key` with which to call this `Op`.
    """

    __uri__ = URI(Op) + "/delete"

    def __init__(self, subject, key=None):
        Op.__init__(self, subject, key)

    def __args__(self):
        return self.subject, self.args

    def __json__(self):
        return {str(URI(self)): to_json([self.subject, self.args])}

    def __repr__(self):
        return f"DELETE {repr(self.subject)}: {repr(self.args)}"

    def __ns__(self, cxt, name_hint):
        from .functions import is_op_ref

        super().__ns__(cxt, name_hint)
        cxt.deanonymize(self.args, name_hint + "_key")

        if is_op_ref(self.args):
            cxt.assign(self.args, name_hint + "_key")
