from ..error import BadRequest
from ..scalar import ref
from ..util import deanonymize, form_of, hex_id, to_json

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.subject}, {self.args})"

    def forward(self):
        """Return the result of evaluating this `Operator`"""

        raise NotImplementedError(f"{self.__class__}.forward")

    def backward(self, variable):
        """
        Return the derivative of this :class:`Operator` (may be a numeric constant or itself an :class:`Operator`).

        If a `variable` is specified, this will be the partial derivative w/r/t the given `variable`.
        """

        raise NotImplementedError(f"{self.__class__}.backward")

    def gradients(self, loss):
        """Return the gradients of the :class:`Variable` s that this :class:`Operator` depends on"""

        raise NotImplementedError(f"{self.__class__}.gradients")


class Unary(Operator):
    def __init__(self, subject):
        Operator.__init__(self, subject, None)

    def __ns__(self, context):
        assert self.args is None

        deanonymize(self.subject, context)

        if ref.is_op_ref(self.subject):
            self.subject = ref.reference(context, self.subject)


class Dual(Operator):
    """A differentiable operator with two arguments"""


class Add(Dual):
    def forward(self):
        return Numeric.add(self.subject, self.args)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return subject + arg

    def gradients(self, loss):
        # TODO: there should be a way to avoid this import (same below)
        from ..ml.optimizer import Variable

        grads = {}

        if isinstance(self.subject, Variable):
            grads.update(self.subject.invert(loss))
        elif isinstance(form_of(self.subject), Operator):
            grads.update(form_of(self.subject).gradients(loss))

        if isinstance(self.args, Variable):
            grads.update(self.args.invert(loss))
        elif isinstance(form_of(self.args), Operator):
            grads.update(form_of(self.args).gradients(loss))

        return grads


class MatMul(Dual):
    def forward(self):
        from ..collection.tensor import einsum
        return einsum("...ij,...jk->ik", [self.subject, self.args])

    def backward(self, variable):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return (subject @ self.args) + (self.subject @ arg)

    def gradients(self, loss):
        from ..ml.optimizer import Variable

        def transpose(matrix):
            return type(matrix)(form=ref.If(
                matrix.ndim == 2,
                matrix.transpose(),
                BadRequest("not a matrix: {{tensor}}", tensor=matrix)))

        grads = {}

        grad = loss @ transpose(self.args)
        if isinstance(self.subject, Variable):
            grads.update(self.subject.invert(grad))
        elif isinstance(form_of(self.subject), Operator):
            grads.update(form_of(self.subject).gradients(grad))

        grad = transpose(self.subject) @ loss
        if isinstance(self.args, Variable):
            grads.update(self.args.invert(grad))
        elif isinstance(form_of(self.args), Operator):
            grads.update(form_of(self.args).gradients(grad))

        return grads


class Mul(Dual):
    def forward(self):
        return Numeric.mul(self.subject, self.args)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return (subject * self.args) + (self.subject * arg)

    def gradients(self, loss):
        from ..ml.optimizer import Variable

        grads = {}

        grad = self.args * loss
        if isinstance(self.subject, Variable):
            grads.update(self.subject.invert(grad))
        elif isinstance(form_of(self.subject), Operator):
            grads.update(form_of(self.subject).gradients(grad))

        grad = self.subject * loss
        if isinstance(self.args, Variable):
            grads.update(self.args.invert(grad))
        elif isinstance(form_of(self.args), Operator):
            grads.update(form_of(self.args).gradients(grad))

        return grads


class Sub(Dual):
    def forward(self):
        return Numeric.sub(self.subject, self.args)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return subject - arg

    def gradients(self, loss):
        from ..ml.optimizer import Variable

        grads = {}

        if isinstance(self.subject, Variable):
            grads.update(self.subject.invert(-loss))
        elif isinstance(form_of(self.subject), Operator):
            grads.update(form_of(self.subject).gradients(loss))

        if isinstance(self.args, Variable):
            grads.update(self.args.invert(-loss))
        elif isinstance(form_of(self.args), Operator):
            grads.update(form_of(self.args).gradients(loss))

        return grads


class Div(Dual):
    def forward(self):
        return Numeric.div(self.subject, self.args)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return ((subject * self.args) - (self.subject, arg)) / (self.args**2)

    def gradients(self, loss):
        from ..ml.optimizer import Variable

        grads = {}

        if isinstance(self.subject, Variable):
            grads.update(self.subject.invert(self.subject * loss / self.args))
        elif isinstance(form_of(self.subject), Operator):
            grads.update(form_of(self.subject).gradients(loss / self.args))

        if isinstance(self.args, Variable):
            grads.update(self.args.invert((-self.subject * loss) / self.args**2))
        elif isinstance(form_of(self.args), Operator):
            grads.update(form_of(self.args).gradients(loss / self.args))

        return grads


class Pow(Dual):
    def forward(self):
        return Numeric.pow(self.subject, self.args)

    def backward(self, variable):
        if self.args is variable:
            return (self.subject**self.args) * self.subject.ln()

        return self.args * (self.subject**(self.args - 1))

    def gradients(self, loss):
        from ..ml.optimizer import Variable

        grads = {}

        if isinstance(self.subject, Variable):
            grads.update(self.subject.invert(loss * self.args * self.subject**(self.args - 1)))
        elif isinstance(form_of(self.subject), Operator):
            grads.update(form_of(self.subject).gradients(loss * self.args * self.subject ** (self.args - 1)))

        if isinstance(self.args, Variable):
            grads.update(self.args.invert(loss * self.subject.log() * self.subject**self.args))
        elif isinstance(form_of(self.args), Operator):
            grads.update(form_of(self.args).gradients(loss * self.subject.log() * self.subject ** self.args))

        return grads


class Exp(Unary):
    def __init__(self, subject):
        Operator.__init__(self, subject, None)

    def forward(self):
        return Numeric.exp(self.subject)

    def backward(self, _variable):
        return self.subject.exp()

    def gradients(self, loss):
        from ..ml.optimizer import Variable

        grad = self.subject.exp() * loss

        if isinstance(self.subject, Variable):
            return self.subject.invert(grad)
        elif isinstance(form_of(self.subject), Operator):
            return form_of(self.subject).gradients(grad)


class Sin(Unary):
    def forward(self):
        return Trigonometric.sin(self.subject)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return subject.cos()


class Cos(Unary):
    def forward(self):
        return Trigonometric.cos(self.subject)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable)
        return subject - self.subject.sin()


class Asin(Unary):
    def forward(self):
        return Trigonometric.asin(self.subject)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return (1 - (subject**2))**-0.5


class Acos(Unary):
    def forward(self):
        return Trigonometric.acos(self.subject)
    
    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return -((1 - subject**2)**-0.5)


class Sinh(Unary):
    def forward(self):
        return Trigonometric.sinh(self.subject)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return subject.cosh()


class Cosh(Unary):
    def forward(self):
        return Trigonometric.cosh(self.subject)
    
    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return subject.sinh()


class Asinh(Unary):
    def forward(self):
        return Trigonometric.asinh(self.subject)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return (subject**2 + 1)**-0.5


class Acosh(Unary):
    def forward(self):
        return Trigonometric.acosh(self.subject)
    
    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return ((subject**2) - 1)**-0.5


class Tan(Unary):
    def forward(self):
        return Trigonometric.tan(self.subject)
    
    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return 1 / (subject.cos()**2)


class Tanh(Unary):
    def forward(self):
        return Trigonometric.tanh(self.subject)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return 1 / subject.cosh()**2


class Atan(Unary):
    def forward(self):
        return Trigonometric.atan(self.subject)

    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return 1 / (subject**2 + 1)


class Atanh(Unary):
    def forward(self):
        return Trigonometric.atanh(self.subject)
    
    def backward(self, variable):
        subject = derivative_of(self.subject, variable) if self.subject is variable else self.subject
        return 1 / (1 - (subject**2))


def derivative_of(state, variable=None):
    """
    Find the derivative of the given `state`.

    If a `variable` is specified, this will be the partial derivative with respect to `variable`.
    If the given `state` is not differentiable, this will raise a `TypeError`.
    """

    from ..scalar.number import Number
    from ..collection.tensor import Dense, Sparse, Tensor
    from ..ml.optimizer import Variable

    if isinstance(state, Variable):
        if variable is None:
            # it's not a partial derivative
            return Dense.ones_like(state)
        elif state is variable:
            # it's a partial derivative, but this is the free variable
            return Dense.ones_like(state)
        else:
            # it's a partial derivative and this variable is held constant
            return Sparse.zeros_like(state)

    if isinstance(form_of(state), Operator):
        return form_of(state).backward(variable)
    elif isinstance(state, Number):
        return type(state)(form=0)
    elif isinstance(state, Tensor):
        return Sparse.zeros_like(state)
    elif isinstance(state, int) or isinstance(state, float):
        return 0
    else:
        raise TypeError(f"the derivative of {state} is not defined")
