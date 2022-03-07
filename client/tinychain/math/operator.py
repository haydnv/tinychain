from ..scalar.ref import is_op_ref, reference, Op
from ..util import deanonymize, form_of, to_json

from .interface import Numeric


class GradientTape(object):
    """A helper class to record gradients as a differentiable operation is constructed"""

    def __init__(self):
        self._tape = []

    def __getitem__(self, i):
        return self._tape[i]

    def append(self, operator):
        self._tape.append(operator)


class Operator(Op):
    """A differentiable operator like addition, multiplication, exponentiation, etc."""

    def __init__(self, subject, arg):
        tape = None

        if isinstance(form_of(subject), Operator):
            tape = form_of(subject).tape

        if isinstance(form_of(arg), Operator):
            if tape and hasattr(arg, "tape") and tape is not form_of(arg).tape:
                raise NotImplementedError("merge two gradient tapes")

            if hasattr(arg, "tape"):
                tape = arg.tape

        self.subject = subject
        self.args = arg
        self.tape = GradientTape() if tape is None else tape
        self.tape.append(self.backward())

    def __json__(self):
        return to_json(self.forward())

    def __ns__(self, context):
        deanonymize(self.subject, context)
        deanonymize(self.args, context)

        if is_op_ref(self.subject):
            self.subject = reference(context, self.subject)

        if is_op_ref(self.args):
            self.args = reference(context, self.args)

    def forward(self):
        """Return the result of evaluating this `Operator`"""

        raise NotImplementedError(f"{self.__class__}.forward")

    def backward(self):
        """Return the derivative of this `Operator` (may be a numeric constant or itself an `Operator`)"""

        raise NotImplementedError(f"{self.__class__}.backward")


class Add(Operator):
    def forward(self):
        return Numeric.add(self.subject, self.args)

    def backward(self):
        subject = _derivative(self.subject)
        arg = _derivative(self.args)

        if subject and arg:
            return Add(subject, arg)
        elif subject:
            return subject
        elif arg:
            return arg
        else:
            return 0


class Mul(Operator):
    def forward(self):
        return Numeric.mul(self.subject, self.args)

    def backward(self):
        subject = _derivative(self.subject)
        arg = _derivative(self.args)

        if subject and arg:
            return Add(Mul(subject, self.args), Mul(self.subject, arg))
        elif subject:
            return Mul(subject, self.args)
        elif arg:
            return Mul(self.subject, arg)
        else:
            return 0


def _derivative(state):
    if isinstance(state, Operator):
        return state.derivative()
    elif isinstance(state, GradientTape):
        return state[-1].derivative()
    else:
        return 0
