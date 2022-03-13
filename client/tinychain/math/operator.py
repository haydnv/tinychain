from ..scalar.ref import is_op_ref, reference, Op
from ..util import deanonymize, form_of, to_json

from .interface import Numeric


class Operator(Op):
    """A differentiable operator like addition, multiplication, exponentiation, etc."""

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
        return Add(subject, arg)


class Mul(Operator):
    def forward(self):
        return Numeric.mul(self.subject, self.args)

    def backward(self):
        subject = _derivative(self.subject)
        arg = _derivative(self.args)
        return Add(Mul(subject, self.args), Mul(self.subject, arg))


def _derivative(state):
    if isinstance(form_of(state), Operator):
        return form_of(state).derivative()
    else:
        from ..collection.tensor import Sparse
        return Sparse.zeros(state.shape, state.dtype)
