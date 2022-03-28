import logging

from ..scalar import ref
from ..state import State
from ..util import deanonymize, form_of, to_json

from .interface import Numeric, Trigonometric


class Operator(ref.Op):
    """A differentiable operator like addition, multiplication, exponentiation, etc."""

    def __init__(self, subject, args):
        if not isinstance(subject, Numeric):
            raise ValueError(f"the subject of a differentiable Operator must implement Numeric (got {subject})")

        ref.Op.__init__(self, subject, args)

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

    def backward(self, variable):
        """
        Return the derivative of this `Operator` (may be a numeric constant or itself an `Operator`).

        If a `variable` is specified, this will be the partial derivative w/r/t the given `variable`.
        """

        raise NotImplementedError(f"{self.__class__}.backward")


class Add(Operator):
    def forward(self):
        return Numeric.add(self.subject, self.args)

    def backward(self, variable):
        from ..hosted_ml.optimizer import Variable
        
        if isinstance(self.args, Variable):
            self.args.grad = variable
        elif isinstance(form_of(self.args), Operator):
            form_of(self.args).backward(variable)
        
        if isinstance(self.subject, Variable):
            self.subject.grad = variable
        elif isinstance(form_of(self.subject), Operator):
            form_of(self.subject).backward(variable)


class Exp(Operator):
    def __init__(self, subject):
        Operator.__init__(self, subject, None)

    def forward(self):
        return Numeric.exp(self.subject)

    def backward(self, variable):
        from ..hosted_ml.optimizer import Variable

        if isinstance(self.subject, Variable):
            self.subject.grad = self.subject.exp() * variable
        elif isinstance(form_of(self.subject), Operator):
            form_of(self.subject).backward(self.subject.exp() * variable)


class MatMul(Operator):
    def forward(self):
        from ..collection.tensor import einsum
        return einsum("...ij,...jk->ik", [self.subject, self.args])

    def backward(self, variable):
        subject = derivative(self.subject, variable)
        arg = derivative(self.args, variable)
        return (subject @ self.args) + (self.subject @ arg)


class Mul(Operator):
    def forward(self):
        return Numeric.mul(self.subject, self.args)

    def backward(self, variable):
        from ..hosted_ml.optimizer import Variable

        if isinstance(self.args, Variable):
            self.args.grad = self.subject*variable
        elif isinstance(form_of(self.args), Operator):
            form_of(self.args).backward(self.subject)*variable
        
        if isinstance(self.subject, Variable):
            self.subject.grad = self.args*variable
        elif isinstance(form_of(self.subject), Operator):
            form_of(self.subject).backward(variable * (self.args))


class Sub(Operator):
    def forward(self):
        return Numeric.sub(self.subject, self.args)

    def backward(self, variable):
        from ..hosted_ml.optimizer import Variable

        if isinstance(self.args, Variable):
            self.args.grad = variable*(-1)
        elif isinstance(form_of(self.args), Operator):
            form_of(self.args).backward(variable)
        
        if isinstance(self.subject, Variable):
            self.subject.grad = variable*(-1)
        elif isinstance(form_of(self.subject), Operator):
            form_of(self.subject).backward(variable)


class Div(Operator):
    def forward(self):
        return Numeric.div(self.subject, self.args)

    def backward(self, variable):
        from ..hosted_ml.optimizer import Variable

        if isinstance(self.args, Variable):
            self.args.grad = (-1)*self.subject*variable/Pow(self.args, 2)
        elif isinstance(form_of(self.args), Operator):
            form_of(self.args).backward(variable / self.args)
        
        if isinstance(self.subject, Variable):
            self.subject.grad = self.subject*variable/self.args
        elif isinstance(form_of(self.subject), Operator):
            form_of(self.subject).backward(variable / self.args)


class Pow(Operator):
    def forward(self):
        return Numeric.pow(self.subject, self.args)

    def backward(self, variable):
        from ..hosted_ml.optimizer import Variable

        if isinstance(self.args, Variable):
            self.args.grad = variable * (self.subject).log() * self.subject**self.args
        elif isinstance(form_of(self.args), Operator):
            form_of(self.args).backward(variable * (self.subject).log() * self.subject**self.args)
        
        if isinstance(self.subject, Variable):
            self.subject.grad = variable * self.args * self.subject**(self.args-1)
        elif isinstance(form_of(self.subject), Operator):
            form_of(self.subject).backward(variable * self.args * self.subject**(self.args-1))


class Unary(Operator):
    def __init__(self, subject):
        Operator.__init__(self, subject, None)


class Sin(Unary):
    def forward(self):
        return Trigonometric.sin(self.subject)

    def backward(self, variable):
        assert not variable  # TODO: partial derivative of trigonometric functions (same below)
        return self.subject.cos()


class Cos(Unary):
    def forward(self):
        return Trigonometric.cos(self.subject)

    def backward(self, variable):
        subject = derivative(self.subject, variable)
        return subject - self.subject.sin()


class Asin(Unary):
    def forward(self):
        return Trigonometric.asin(self.subject)

    def backward(self, variable):
        assert not variable
        return (1 - (self.subject**2))**-0.5


class Acos(Unary):
    def forward(self):
        return Trigonometric.acos(self.subject)
    
    def backward(self, variable):
        assert not variable
        return -((1 - self.subject**2)**-0.5)


class Sinh(Unary):
    def forward(self):
        return Trigonometric.sinh(self.subject)

    def backward(self, variable):
        assert not variable
        return self.subject.cosh()


class Cosh(Unary):
    def forward(self):
        return Trigonometric.cosh(self.subject)
    
    def backward(self, variable):
        assert not variable
        return self.subject.sinh()


class Asinh(Unary):
    def forward(self):
        return Trigonometric.asinh(self.subject)

    def backward(self, variable):
        assert not variable
        return (self.subject**2 + 1)**-0.5


class Acosh(Unary):
    def forward(self):
        return Trigonometric.acosh(self.subject)
    
    def backward(self, variable):
        assert not variable
        return ((self.subject**2) - 1)**-0.5


class Tan(Unary):
    def forward(self):
        return Trigonometric.tan(self.subject)
    
    def backward(self, variable):
        assert not variable
        return 1 / (self.subject.cos()**2)


class Tanh(Unary):
    def forward(self):
        return Trigonometric.tanh(self.subject)
    
    def backward(self, variable):
        assert not variable
        return 1 / self.subject.cosh()**2


class Atan(Unary):
    def forward(self):
        return Trigonometric.atan(self.subject)

    def backward(self, variable):
        assert not variable
        return 1 / (self.subject**2 + 1)


class Atanh(Unary):
    def forward(self):
        return Trigonometric.atanh(self.subject)
    
    def backward(self, variable):
        assert not variable
        return 1 / (1 - (self.subject**2))


def derivative(state, variable=None):
    from ..scalar.number import Number
    from ..collection.tensor import Dense, Sparse, Tensor
    from ..hosted_ml.optimizer import Variable

    if isinstance(state, Variable):
        if variable is None:
            # it's not a partial derivative
            return Dense.ones_like(state)
        elif state is variable:
            # it's a partial derivative, but this is the free variable
            return Dense.ones_like(state)
        else:
            # it's a partial derivative and this variable is held constant
            return Dense.zeros_like(state)

    if isinstance(form_of(state), Operator):
        return form_of(state).backward(variable)
    elif isinstance(state, Number):
        return type(state)(0)
    elif isinstance(state, Tensor):
        return Sparse.zeros_like(state)
    elif isinstance(state, int) or isinstance(state, float):
        return 0
    else:
        raise TypeError(f"the derivative of {state} is not defined")
