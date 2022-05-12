import logging
import typing

from ..error import BadRequest
from ..scalar.ref import deref, hex_id, is_literal, same_as, is_op_ref, reference, If, Op
from ..scalar.value import Id
from ..util import deanonymize, to_json

from .base import is_numeric
from .interface import Numeric, Trigonometric


class Gradients(dict):
    def __setitem__(self, key: Id, value: Numeric):
        if key in self:
            dict.__setitem__(self, key, self[key] + value)
        else:
            dict.__setitem__(self, key, value)

    def update(self, __m: typing.Mapping[Id, Numeric], **kwargs: Numeric) -> None:
        for var_id in __m:
            self[var_id] = __m[var_id]

        for var_id in kwargs:
            self[var_id] = __m[var_id]


class Operator(Op):
    """A differentiable operator like addition, multiplication, exponentiation, etc."""

    def __init__(self, subject, args):
        if not is_numeric(subject):
            logging.info(f"{subject} is the the subject of a differentiable Operator but does not implement Numeric")

        Op.__init__(self, simplify(subject), simplify(args))

    def __json__(self):
        return to_json(self.forward())

    def __ns__(self, context):
        deanonymize(self.subject, context)
        deanonymize(self.args, context)

        if is_op_ref(self.subject):
            self.subject = reference(context, self.subject)

        if is_op_ref(self.args):
            self.args = reference(context, self.args)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.subject}, {self.args})"

    @property
    def shape(self):
        raise NotImplementedError(f"{self.__class__.__name__}.shape")

    def forward(self):
        """Return the result of evaluating this `Operator`"""

        raise NotImplementedError(f"{self.__class__}.forward")

    def backward(self, variable=None):
        """
        Return the derivative of this :class:`Operator` (may be a numeric constant or itself an :class:`Operator`).

        If a `variable` is specified, this will be the partial derivative w/r/t the given `variable`.
        """

        raise NotImplementedError(f"{self.__class__}.backward")

    # TODO: support computing gradients w/r/t arbitrary states in the operator graph
    def gradients(self, loss):
        """
        Return the :class:`Gradients` this :class:`Operator` with respect to the given `variables`.

        If no `variables` are specified, this will return the `Gradients` of each :class:`Variable` that this
        `Operator` depends on.
        """

        raise NotImplementedError(f"{self.__class__}.gradients")

    def simplify(self):
        """
        Return a simplified form of this :class:`Operator` if possible, otherwise return `self`.

        For example, calling `Add(x, 0).simplify()` will return `x`, and `Mul(1, y).simplify()` will return `y`.
        """

        return self


class Unary(Operator):
    def __init__(self, subject):
        if not is_numeric(subject):
            raise ValueError(f"Unary operator requires a Numeric subject, not {subject}")

        Operator.__init__(self, subject, None)

    def __ns__(self, context):
        assert self.args is None

        deanonymize(self.subject, context)

        if is_op_ref(self.subject):
            self.subject = reference(context, self.subject)


class Custom(Unary):
    """A custom operator"""

    def __init__(self, subject):
        Operator.__init__(self, subject, None)
        self._op = self.forward()

    def __json__(self):
        return to_json(self._op)

    def __ns__(self, context):
        Unary.__ns__(self, context)
        deanonymize(self._op, context)

    @property
    def shape(self):
        return self._op.shape


# TODO: Tensor.log(base!=None)
class Abs(Unary):
    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Numeric.abs(self.subject)

    def backward(self, _variable=None):
        return self.subject / self.subject.abs()

    def gradients(self, loss):
        loss *= self.backward()

        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients())
        else:
            grads[hex_id(self.subject)] = loss

        return grads


class Exp(Unary):
    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Numeric.exp(self.subject)

    def backward(self, _variable=None):
        return self.subject.exp()

    def gradients(self, loss):
        loss *= self.backward()
        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients(loss))
        else:
            grads[hex_id(self.subject)] = loss

        return grads


class Trig(Unary):
    @property
    def shape(self):
        return self.subject.shape

    def gradients(self, loss):
        if operator(self.subject):
            return operator(self.subject).gradients(loss)
        else:
            return Gradients({hex_id(self.subject): loss})


class Sin(Trig):
    def forward(self):
        return Trigonometric.sin(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return subject.cos()

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Cos(Trig):
    def forward(self):
        return Trigonometric.cos(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        return subject - self.subject.sin()

    def gradients(self, loss):
        loss *= -self.subject.sin()
        return Trig.gradients(self, loss)


class Asin(Trig):
    def forward(self):
        return Trigonometric.asin(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return (1 - (subject**2))**-0.5

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Acos(Trig):
    def forward(self):
        return Trigonometric.acos(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return -((1 - subject**2)**-0.5)

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Sinh(Trig):
    def forward(self):
        return Trigonometric.sinh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return subject.cosh()

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Cosh(Trig):
    def forward(self):
        return Trigonometric.cosh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return subject.sinh()

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Asinh(Trig):
    def forward(self):
        return Trigonometric.asinh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return (subject**2 + 1)**-0.5

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Acosh(Trig):
    def forward(self):
        return Trigonometric.acosh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return ((subject**2) - 1)**-0.5

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Tan(Trig):
    def forward(self):
        return Trigonometric.tan(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return 1 / (subject.cos()**2)

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Tanh(Trig):
    def forward(self):
        return Trigonometric.tanh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return 1 - subject.tanh()**2

    def gradients(self, loss):
        loss *= self.backward()
        return Trig.gradients(self, loss)


class Atan(Trig):
    def forward(self):
        return Trigonometric.atan(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return 1 / (subject**2 + 1)

    def gradients(self, loss):
        loss *= (self.subject**2 + 1)**(-1)
        return Trig.gradients(self, loss)


class Atanh(Trig):
    def forward(self):
        return Trigonometric.atanh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return 1 / (1 - (subject**2))

    def gradients(self, loss):
        loss *= (1 - self.subject**2)**(-1)
        return Trig.gradients(self, loss)


class Dual(Operator):
    """A differentiable operator with two arguments"""

    def __init__(self, subject, args):
        if not is_numeric(subject):
            raise ValueError(f"{self.__class__.__name__} requires a Numeric subject, not {subject}")

        if not is_numeric(args):
            raise ValueError(f"{self.__class__.__name__} requires Numeric args, not {args}")

        Operator.__init__(self, subject, args)


#TODO: Tensor.log(base!=None)
class Log(Operator):
    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Numeric.log(self.subject, self.args)

    def backward(self, _variable=None):
        return 1 / self.subject

    def gradients(self, loss):
        loss *= self.backward()

        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients(loss))
        else:
            grads[hex_id(self.subject)] = loss

        return grads


class MatMul(Dual):
    @property
    def shape(self):
        from ..shape import Shape
        return self.subject.shape[:-2] + Shape((self.subject.shape[-1], self.args.shape[-2]))

    def forward(self):
        from ..collection.tensor import einsum
        return einsum("...ij,...jk->ik", [self.subject, self.args])

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return (subject @ self.args) + (self.subject @ arg)

    def gradients(self, loss):
        def transpose(matrix):
            return type(matrix)(form=If(
                matrix.ndim == 2,
                matrix.transpose(),
                BadRequest("not a matrix: {{tensor}}", tensor=matrix)))

        grads = Gradients()

        grad = loss @ transpose(self.args)
        if operator(self.subject):
            grads.update(operator(self.subject).gradients(grad))
        else:
            grads[hex_id(self.subject)] = grad

        grad = transpose(self.subject) @ loss
        if operator(self.args):
            grads.update(operator(self.args).gradients(grad))
        else:
            grads[hex_id(self.args)] = grad

        return grads


class Pow(Dual):
    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Numeric.pow(self.subject, self.args)

    def backward(self, variable=None):
        if self.args is variable:
            return (self.subject**self.args) * self.subject.log()

        return self.args * (self.subject**(self.args - 1))

    def gradients(self, loss):
        grads = Gradients()

        grad = loss * self.args * self.subject**(self.args - 1)
        if operator(self.subject):
            grads.update(operator(self.subject).gradients(grad))
        else:
            grads[hex_id(self.subject)] = grad

        grad = loss * self.subject.log() * self.subject**self.args
        if operator(self.args):
            grads.update(operator(self.args).gradients(grad))
        else:
            grads[hex_id(self.args)] = grad

        return grads

    def simplify(self):
        from ..collection.tensor import Dense, Sparse, NDArray

        if same_as(self.subject, 1):
            return self.subject
        elif isinstance(self.subject, NDArray) and same_as(self.subject, Dense.ones_like(self.subject)):
            if isinstance(self.args, NDArray):
                return self.subject.broadcast(self.args.shape)
            else:
                return self.subject

        if same_as(self.args, 1):
            return self.subject
        elif same_as(self.args, 0):
            return Dense.ones_like(self.subject) if isinstance(self.subject, NDArray) else 1
        elif isinstance(self.args, NDArray) and same_as(self.args, Dense.ones_like(self.args)):
            return self.subject.broadcast(self.args.shape)
        elif isinstance(self.args, NDArray) and same_as(self.args, Sparse.zeros_like(self.args)):
            if isinstance(self.subject, NDArray):
                return Dense.ones_like(self.subject).broadcast(self.args.shape)
            else:
                return Dense.ones(self.args)

        return self


class DualBroadcast(Operator):
    @property
    def shape(self):
        if is_literal(self.args):
            # it's a constant number
            return self.subject.shape

        return self.subject.shape.broadcast(self.args.shape)


class Add(DualBroadcast):
    def forward(self):
        return Numeric.add(self.subject, self.args)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return subject + arg

    def gradients(self, loss):
        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients(loss))
        else:
            grads[hex_id(self.subject)] = loss

        if operator(self.args):
            grads.update(operator(self.args).gradients(loss))
        else:
            grads[hex_id(self.args)] = loss

        return grads

    def simplify(self):
        from ..collection.tensor import NDArray, Sparse

        if is_literal(self.args):
            if same_as(self.args, 0):
                return self.subject
        elif isinstance(self.subject, NDArray) and same_as(self.args, Sparse.zeros_like(self.args)):
            return self.subject.broadcast(self.args.shape)

        if is_literal(self.subject):
            if same_as(self.subject, 0):
                return self.args
        elif isinstance(self.args, NDArray) and same_as(self.subject, Sparse.zeros_like(self.subject)):
            return self.args.broadcast(self.subject.shape)

        return self


class Mul(DualBroadcast):
    def forward(self):
        return Numeric.mul(self.subject, self.args)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return (subject * self.args) + (self.subject * arg)

    def gradients(self, loss):
        grads = Gradients()

        grad = self.args * loss
        if operator(self.subject):
            grads.update(operator(self.subject).gradients(grad))
        else:
            grads[hex_id(self.subject)] = grad

        grad = self.subject * loss
        if operator(self.args):
            grads.update(operator(self.args).gradients(grad))
        else:
            grads[hex_id(self.args)] = grad

        return grads

    def simplify(self):
        from ..collection.tensor import Dense, Sparse, NDArray

        if is_literal(self.args):
            if same_as(self.args, 0):
                return 0
            elif same_as(self.args, 1):
                return self.subject
        elif same_as(self.args, Sparse.zeros_like(self.args)):
            return self.args.broadcast(self.subject.shape)
        elif isinstance(self.subject, NDArray) and same_as(self.args, Dense.ones_like(self.args)):
            return self.subject.broadcast(self.args.shape)

        if is_literal(self.subject):
            if same_as(self.subject, 0):
                return 0
            elif same_as(self.subject, 1):
                return self.args
        elif same_as(self.subject, Sparse.zeros_like(self.subject)):
            return self.subject.broadcast(self.args.shape)
        elif isinstance(self.args, NDArray) and same_as(self.subject, Dense.ones_like(self.subject)):
            return self.args.broadcast(self.subject.shape)

        return self


class Sub(DualBroadcast):
    def forward(self):
        return Numeric.sub(self.subject, self.args)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return subject - arg

    def gradients(self, loss):
        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients(loss))
        else:
            grads[hex_id(self.subject)] = -loss

        if operator(self.args):
            grads.update(operator(self.args).gradients(loss))
        else:
            grads[hex_id(self.args)] = -loss

        return grads

    def simplify(self):
        from ..collection.tensor import Sparse

        if is_literal(self.args):
            if same_as(self.args, 0):
                return self.subject
        elif same_as(self.args, Sparse.zeros_like(self.args)):
            return self.subject

        return self


class Div(DualBroadcast):
    def __init__(self, subject, args):
        if same_as(args, 0):
            raise ValueError(f"cannot divide {subject} by {args}")

        DualBroadcast.__init__(self, subject, args)

    def forward(self):
        return Numeric.div(self.subject, self.args)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)

        return ((subject * self.args) - (self.subject * arg)) / (self.args**2)

    def gradients(self, loss):
        grads = Gradients()

        if operator(self.subject):
            grads.update(operator(self.subject).gradients(loss / self.args))
        else:
            grads[hex_id(self.subject)] = self.subject * loss / self.args

        if operator(self.args):
            grads.update(operator(self.args).gradients(loss / self.args))
        else:
            grads[hex_id(self.args)] = (-self.subject * loss) / self.args**2

        return grads

    def simplify(self):
        from ..collection.tensor import Dense, NDArray

        if is_literal(self.subject):
            if same_as(self.subject, 0):
                return 0
        elif isinstance(self.args, NDArray) and same_as(self.subject, Dense.zeros_like(self.subject)):
            return self.args.broadcast(self.subject.shape)

        if is_literal(self.args):
            if same_as(self.args, 1):
                return self.subject
        elif isinstance(self.subject, NDArray) and same_as(self.args, Dense.ones_like(self.args)):
            return self.subject.broadcast(self.args.shape)

        # TODO: simplify both numerator and denominator into a product of multiplicands

        if operator(self.subject) and isinstance(operator(self.subject), Mul):
            numerator = operator(self.subject)
            if same_as(numerator.subject, self.args):
                return numerator.args
            elif same_as(numerator.args, self.args):
                return numerator.subject

        if operator(self.args) and isinstance(operator(self.args), Mul):
            denominator = operator(self.args)
            if same_as(denominator.subject, self.subject):
                return 1 / denominator.args
            elif same_as(denominator.args, self.subject):
                return 1 / denominator.subject

        return self


def derivative_of(state, variable=None):
    """
    Find the derivative of the given `state`.

    If a `variable` is specified, this will be the partial derivative with respect to `variable`.
    If the given `state` is not differentiable, this will raise a `TypeError`.
    """

    if not is_numeric(state):
        raise ValueError(f"cannot take the derivative of a non-numeric state {state} (note the type {type(state)})")

    from ..collection.tensor import Dense, Sparse

    def ones_like(state):
        if is_literal(state) or same_as(state.shape.ndim(), 0):
            return 1.
        else:
            return Dense.ones_like(state)

    def zeros_like(state):
        if is_literal(state) or same_as(state.shape.ndim(), 0):
            return 0.
        else:
            return Sparse.zeros_like(state)

    if same_as(state, variable):
        # it's a partial derivative and this is the free variable
        return ones_like(state)

    from ..ml.optimizer import Variable

    if isinstance(state, Variable):
        if variable is None:
            # it's not a partial derivative
            return ones_like(state)
        else:
            # it's a partial derivative and this variable is held constant
            return zeros_like(state)

    if operator(state):
        return operator(state).backward(variable)

    return zeros_like(state)


def gradients(differentiable, loss, variables=None):
    """
    Return the gradient of a `differentiable` operator graph with respect the `loss`.

    If one variable is given, one gradient will be returned, or a `KeyError` will be raised if not present in the graph.
    If a list of variables is given, a corresponding list of gradients will be returned.
    If no variables are given, a :class:`Gradients` object whose keys are the `hex_id` of each input.
    """

    from .equation import Function

    if not operator(differentiable):
        raise ValueError(f"not a differentiable state: {differentiable}")

    grads = operator(differentiable).gradients(loss)

    if variables is None:
        return Function(grads).optimize()

    if not isinstance(variables, (list, tuple)):
        if hex_id(variables) not in grads:
            raise KeyError(f"{variables} is not reachable from operator {differentiable}")

        return Function(grads[hex_id(variables)]).optimize()

    missing = [var for var in variables if hex_id(var) not in grads]
    if missing:
        raise KeyError(f"not reachable by traversing the operator graph {differentiable}: {missing}")

    return Function([grads[hex_id(var)] for var in variables]).optimize()


def operator(state_or_ref):
    """Return the `Operator` instance which produces the given `state_or_ref`, if any"""

    if isinstance(state_or_ref, Operator):
        return state_or_ref
    elif deref(state_or_ref) is not state_or_ref:
        return operator(deref(state_or_ref))


def simplify(state):
    """
    Simplify the given operator graph, if possible.

    For example, `simplify(Add(0, 2))` will return `2`.
    """

    while operator(state):
        simplified = operator(state).simplify()
        if same_as(simplified, state):
            break

        state = simplified

    return state
