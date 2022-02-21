"""
Reference types.
:class:`After`, :class:`Case`, :class:`If`, and :class:`While` are available in the top-level namespace.
"""

import logging

from ..reflect import is_conditional, is_ref
from ..util import deanonymize, form_of, get_ref, hex_id, to_json, uri, URI


class Ref(object):
    """A reference to a :class:`State`. Prefer to construct a subclass like :class:`If` or :class:`Get`."""

    __uri__ = URI("/state/scalar/ref")


class After(Ref):
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
        if is_conditional(then):
            raise ValueError(f"After does not support a conditional clause: {then}")

        self.when = when
        self.then = then

    def __dbg__(self):
        return [self.when, self.then]

    def __json__(self):
        return {str(uri(self)): to_json([self.when, self.then])}

    def __ns__(self, cxt):
        deanonymize(self.when, cxt)
        deanonymize(self.then, cxt)

        if is_conditional(self.when):
            self.when = reference(cxt, self.when)


class Case(Ref):
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

    def __dbg__(self):
        return [self.cond, self.switch, self.case]

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.switch, self.case])}

    def __ns__(self, cxt):
        deanonymize(self.cond, cxt)
        deanonymize(self.switch, cxt)
        deanonymize(self.case, cxt)


class If(Ref):
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
        if is_conditional(cond):
            raise ValueError(f"If does not support nested conditionals: {cond}")

        self.cond = cond
        self.then = then
        self.or_else = or_else

    def __dbg__(self):
        return [self.cond, self.then, self.or_else]

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.then, self.or_else])}

    def __ns__(self, cxt):
        deanonymize(self.cond, cxt)
        deanonymize(self.then, cxt)
        deanonymize(self.or_else, cxt)

        if is_conditional(self.cond) or is_op_ref(self.cond):
            self.cond = reference(cxt, self.cond)


class While(Ref):
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

    def __dbg__(self):
        return [self.cond, self.op, self.state]

    def __json__(self):
        return {str(uri(self)): to_json([self.cond, self.op, self.state])}

    def __ns__(self, cxt):
        deanonymize(self.cond, cxt)
        deanonymize(self.op, cxt)
        deanonymize(self.state, cxt)


class With(Ref):
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

    def __dbg__(self):
        return [self.capture, self.op]

    def __json__(self):
        return {str(uri(self)): to_json([self.capture, self.op])}

    def __ns__(self, _cxt):
        pass

    def __ref__(self, name):
        return get_ref(self.op, name)


class Op(Ref):
    """A resolvable reference to an :class:`Op`."""

    __uri__ = uri(Ref) + "/op"

    def __init__(self, subject, args):
        self.subject = subject
        self.args = args

    def __dbg__(self):
        subject = [self.subject] if is_ref(self.subject) else []
        return subject + list(self.args)

    def __json__(self):
        if hasattr(self.subject, "__form__"):
            subject = form_of(self.subject)
        else:
            subject = self.subject

        if uri(subject) is None:
            raise ValueError(f"Op subject {subject} has no URI")

        return {str(uri(subject)): to_json(self.args)}

    def __ns__(self, cxt):
        deanonymize(self.subject, cxt)


class Get(Op):
    """
    A `Get` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        key (Value or Ref): The `key` with which to call this `Op`.
    """

    __uri__ = uri(Op) + "/get"

    def __init__(self, subject, key=None):
        if subject is None:
            raise ValueError("Get op ref subject cannot be None")

        Op.__init__(self, subject, (key,))

    def __json__(self):
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
        return f"GET Op ref {self.subject} {self.args}"

    def __ns__(self, cxt):
        super().__ns__(cxt)

        deanonymize(self.args, cxt)

        if is_op_ref(self.args):
            (key,) = self.args
            log_anonymous(key)
            key = reference(cxt, key)
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
        return f"PUT Op ref {self.subject} {self.args}"

    def __ns__(self, cxt):
        super().__ns__(cxt)

        key, value = self.args
        deanonymize(key, cxt)
        deanonymize(value, cxt)

        if is_op_ref(key):
            log_anonymous(key)
            key = reference(cxt, key)

        if is_op_ref(value):
            log_anonymous(value)
            value = reference(cxt, value)

        self.args = (key, value)


class Post(Op):
    """
    A `Post` :class:`Op` reference to resolve.

    Args:
        subject (State or URI): The instance of which this `Op` is a method (can be a `URI`).

        args (Map or Ref): The parameters with which to call this `Op`.
    """

    __uri__ = uri(Op) + "/post"

    def __init__(self, subject, args):
        if not hasattr(args, "__iter__"):
            raise ValueError("POST Op ref requires named parameters (try using a Python dict)")

        Op.__init__(self, subject, args)

    def __repr__(self):
        return f"POST Op ref {self.subject} {self.args}"

    def __ns__(self, cxt):
        super().__ns__(cxt)

        if not isinstance(self.args, dict):
            raise ValueError(f"POST arguments must be a Python dict, not {self.args}")

        args = {}
        for name, arg in self.args.items():
            deanonymize(arg, cxt)

            if is_op_ref(arg):
                log_anonymous(arg)
                args[name] = reference(cxt, arg)
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
        return f"DELETE Op ref {self.subject} {self.args}"

    def __ns__(self, cxt):
        super().__ns__(cxt)
        deanonymize(self.args, cxt)

        if is_op_ref(self.args):
            log_anonymous(self.args)
            self.args = reference(cxt, self.args)


class MethodSubject(object):
    def __init__(self, subject, method_name=""):
        if uri(subject).startswith('$'):
            self.__uri__ = uri(subject).append(method_name)
        else:
            self.__uri__ = None

        self.subject = subject
        self.method_name = method_name

    def __dbg__(self):
        subject = [self.subject] if is_ref(self.subject) else []
        return subject + self.args

    def __dbg__(self):
        return [self.subject]

    def __ns__(self, cxt):
        name = f"{self.subject.__class__.__name__}_{hex_id(self.subject)}"
        auto_uri = URI(name).append(self.method_name)

        if uri(self):
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

            deanonymize(self.subject, cxt)

        else:
            deanonymize(self.subject, cxt)
            self.__uri__ = uri(self.subject).append(self.method_name)


    def __json__(self):
        if self.__uri__ is None:
            raise ValueError(f"cannot call method {self.method_name} on an anonymous subject {self.subject}")

        return to_json(uri(self))

    def __str__(self):
        if self.__uri__ is None:
            return str(uri(self.subject).append(self.method_name))
        else:
            return str(uri(self))

    def __repr__(self):
        return f"MethodSubject {repr(self.subject)}"


def is_op_ref(fn):
    if isinstance(fn, Op):
        return True
    elif hasattr(fn, "__form__"):
        return is_op_ref(form_of(fn))
    elif isinstance(fn, list) or isinstance(fn, tuple):
        return any(is_op_ref(item) for item in fn)
    elif isinstance(fn, dict):
        return any(is_op_ref(fn[k]) for k in fn)
    else:
        return False


def is_write_op_ref(fn):
    if isinstance(fn, Delete) or isinstance(fn, Put):
        return True
    elif hasattr(fn, "__form__"):
        return is_write_op_ref(form_of(fn))
    elif isinstance(fn, list) or isinstance(fn, tuple):
        return any(is_write_op_ref(item) for item in fn)
    elif isinstance(fn, dict):
        return any(is_write_op_ref(fn[k]) for k in fn)
    else:
        return False


def log_anonymous(arg):
    if is_write_op_ref(arg):
        logging.warning(f"assigning auto-generated name to the result of {arg}")
    elif is_op_ref(arg):
        logging.info(f"assigning auto-generated name to the result of {arg}")


def reference(cxt, state):
    name = f"{state.__class__.__name__}_{hex_id(state)}"
    if name not in cxt:
        logging.debug(f"assigned name {name} to {state} in {cxt}")
        setattr(cxt, name, state)

    return getattr(cxt, name)
