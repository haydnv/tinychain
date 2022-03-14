from ..scalar import ref
from ..util import deanonymize, form_of, to_json

from .interface import Numeric


class Operator(ref.Op):
    """A differentiable operator like addition, multiplication, exponentiation, etc."""

    def __json__(self):
        return to_json(self.forward())

    def __ns__(self, context):
        deanonymize(self.subject, context)
        deanonymize(self.args, context)

        if ref.is_op_ref(self.subject):
            self.subject = ref.reference(context, self.subject)

        if ref.is_op_ref(self.args):
            self.args = ref.reference(context, self.args)

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


class Exp(Operator):
    def __init__(self, subject):
        Operator.__init__(self, subject, None)

    def forward(self):
        return Numeric.exp(self.subject)

    def backward(self):
        return self


class MatMul(Operator):
    def forward(self):
        from ..collection.tensor import einsum
        return einsum("...ij,...jk->ik", [self.subject, self.args])

    def backward(self):
        subject = _derivative(self.subject)
        arg = _derivative(self.arg)
        return Add(MatMul(subject, self.arg), MatMul(self.subject, arg))


class Mul(Operator):
    def forward(self):
        return Numeric.mul(self.subject, self.args)

    def backward(self):
        subject = _derivative(self.subject)
        arg = _derivative(self.args)
        return Add(Mul(subject, self.args), Mul(self.subject, arg))


class Sub(Operator):
    def forward(self):
        return Numeric.sub(self.subject, self.args)

    def backward(self):
        subject = _derivative(self.subject)
        arg = _derivative(self.arg)
        return Sub(subject, arg)


class Div(Operator):
    def forward(self):
        return Numeric.div(self.subject, self.args)

    def backward(self):
        subject = _derivative(self.subject)
        arg = _derivative(self.arg)
        return Div(Sub(Mul(subject, self.arg), Mul(self.subject, arg)), Pow(self.arg, 2))


class Pow(Operator):
    def forward(self):
        return Numeric.pow(self.subject, self.args)

    def backward(self):
        return Mul(self.arg, Pow(self.subject, Sub(self.arg, 1)))


def _derivative(state):
    if isinstance(form_of(state), Operator):
        return form_of(state).derivative()
    else:
        from ..collection.tensor import Sparse
        return Sparse.zeros(state.shape, state.dtype)
