from ..scalar import ref
from ..util import deanonymize, form_of, to_json

from .interface import Numeric, Trigonometric


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
        arg = _derivative(self.args)
        return Add(MatMul(subject, self.args), MatMul(self.subject, arg))


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
        arg = _derivative(self.args)
        return Sub(subject, arg)


class Div(Operator):
    def forward(self):
        return Numeric.div(self.subject, self.args)

    def backward(self):
        subject = _derivative(self.subject)
        arg = _derivative(self.args)
        return Div(Sub(Mul(subject, self.args), Mul(self.subject, arg)), Pow(self.args, 2))


class Pow(Operator):
    def forward(self):
        return Numeric.pow(self.subject, self.args)

    def backward(self):
        return Mul(self.args, Pow(self.subject, Sub(self.args, 1)))


class Unary(Operator):
  def __init__(self, subject):
    Operator.__init__(self, subject, None)


class Sin(Unary):
    def forward(self):
        return Trigonometric.sin(self.subject)

    def backward(self):
        return Cos(self.subject)


class Cos(Unary):
    def forward(self):
        return Trigonometric.cos(self.subject)

    def backward(self):
        subject = _derivative(self.subject)
        return Sub(subject, Sin(self.subject))


class Asin(Unary):
    def forward(self):
        return Trigonometric.asin(self.subject)

    def backward(self):
        from ..collection.tensor import Dense, Tensor
        return Pow(Tensor(Sub(Dense.ones(self.subject.shape), Pow(self.subject, 2))), -0.5)


class Acos(Unary):
    def forward(self):
        return Trigonometric.acos(self.subject)
    
    def backward(self):
        from ..collection.tensor import Dense, Tensor
        return Sub(Dense.zeros(self.subject.shape), Pow(Tensor(Sub(Dense.ones(self.subject.shape), Pow(self.subject, 2))), -0.5))


class Sinh(Unary):
    def forward(self):
        return Trigonometric.sinh(self.subject)

    def backward(self):
        return Cosh(self.subject)


class Cosh(Unary):
    def forward(self):
        return Trigonometric.cosh(self.subject)
    
    def backward(self):
        return Sinh(self.subject)


class Asinh(Unary):
    def forward(self):
        return Trigonometric.asinh(self.subject)

    def backward(self):
        from ..collection.tensor import Dense, Tensor
        return Pow(Tensor(Add(Dense.ones(self.subject.shape), Pow(self.subject, 2))), -0.5)


class Acosh(Unary):
    def forward(self):
        return Trigonometric.acosh(self.subject)
    
    def backward(self):
        from ..collection.tensor import Dense, Tensor
        return Pow(Tensor(Sub(Tensor(Pow(self.subject, 2)), Dense.ones(self.subject.shape))), -0.5)


class Tan(Unary):
    def forward(self):
        return Trigonometric.tan(self.subject)
    
    def backward(self):
        return Div(1, Pow(Cos(self.subject), 2))


def _derivative(state):
    from ..scalar.number import Number
    from ..collection.tensor import Dense, Sparse, Tensor
    from ..hosted_ml.optimizer import Variable

    if isinstance(form_of(state), Operator):
        return form_of(state).backward()
    elif isinstance(state, Variable):
        return Dense.ones_like(state)
    elif isinstance(state, Number):
        return type(state)(0)
    elif isinstance(state, Tensor):
        return Sparse.zeros_like(state)
    else:
        raise TypeError(f"the derivative of {state} is not defined")
