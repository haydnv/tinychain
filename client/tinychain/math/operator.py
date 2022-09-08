import logging
import typing

from ..json import to_json
from ..scalar.ref import deref, is_literal, same_as, is_op_ref, Op
from ..scalar.value import Id

from .base import is_numeric
from .interface import Boolean, Numeric, Trigonometric


class Gradients(dict):
    def __add__(self, other):
        grads = Gradients()
        grads.update(self)
        grads.update(other)
        return grads

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

        Op.__init__(self, subject, args)

    def __args__(self):
        return self.subject, self.args

    def __json__(self):
        return to_json(self.forward())

    def __ns__(self, cxt, name_hint):
        cxt.deanonymize(self.subject, name_hint + "_subject")
        cxt.deanonymize(self.args, name_hint + "_args")

        if is_literal(self.subject) or is_op_ref(self.subject):
            cxt.assign(self.subject, name_hint + "_subject")

        if is_op_ref(self.args):
            cxt.assign(self.args, name_hint + "_args")

    def __repr__(self):
        raise NotImplementedError(f"human-readable string representation of {self.__class__.__name__}")

    def __same__(self, other):
        other = operator(other)

        if not other:
            return False

        return type(self) is type(other) and same_as(self.subject, other.subject) and same_as(self.args, other.args)

    @property
    def shape(self):
        """The shape of the result of this operation"""

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

    def gradients(self, loss):
        """
        Return the :class:`Gradients` this :class:`Operator` with respect to the given `variables`.

        If no `variables` are specified, this will return the `Gradients` of each :class:`Variable` that this
        `Operator` depends on.
        """

        raise NotImplementedError(f"{self.__class__}.gradients")

    def simplify(self):
        """
        Return a simplified but logically equivalent version of this operator, if possible.
        For example, `Mul(2, 1).simplify()` will return 2.

        IMPORTANT: don't call `simplify` until after constructing an entire operator graph.
        This is because `simplify` may discard parts of the operator graph needed to apply the chain rule correctly.
        """

        return self


class Unary(Operator):
    def __init__(self, subject):
        if not is_numeric(subject):
            raise ValueError(f"Unary operator requires a Numeric subject, not {subject}")

        Operator.__init__(self, subject, None)

    def __args__(self):
        return self.subject,

    def __ns__(self, cxt, name_hint):
        assert self.args is None

        cxt.deanonymize(self.subject, name_hint + "_subject")

        if is_op_ref(self.subject):
            cxt.assign(self.subject, name_hint + "_subject")


class Custom(Unary):
    """A custom operator"""

    def __init__(self, subject):
        Operator.__init__(self, subject, None)
        self._op = self.forward()

    def __json__(self):
        return to_json(self._op)

    def __ns__(self, cxt, name_hint):
        Unary.__ns__(self, cxt, name_hint)
        cxt.deanonymize(self._op, name_hint + "_custom_op")

    @property
    def shape(self):
        return self._op.shape


# TODO: Tensor.log(base!=None)
class Abs(Unary):
    def __repr__(self):
        return f"abs({self.subject})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Numeric.abs(self.subject)

    def backward(self, _variable=None):
        return self.subject / self.subject.abs()

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())

    def simplify(self):
        subject = simplify(self.subject)
        return Abs(subject)


class Exp(Unary):
    def __repr__(self):
        return f"e**({self.subject})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Numeric.exp(self.subject)

    def backward(self, variable=None):
        if same_as(variable, self.subject):
            return derivative_of(self.subject, variable).exp()
        elif operator(self.subject):
            # this operator always has exactly one argument, so it's safe to hard-code the single-variable chain rule
            return derivative_of(self.subject) * self.subject.exp()
        else:
            return self.subject.exp()

    def gradients(self, loss):
        return gradients(self.subject, loss * self.subject.exp())

    def simplify(self):
        subject = simplify(self.subject)

        if is_one(subject):
            return 1
        elif is_zero(subject):
            return 0
        else:
            return Exp(subject)


class LogicalNot(Unary):
    def __repr__(self):
        return f"NOT ({self.subject})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Boolean.logical_not(self.subject)

    def simplify(self):
        subject = simplify(self.subject)
        return LogicalNot(subject)


class Trig(Unary):
    @property
    def shape(self):
        return self.subject.shape

    def simplify(self):
        subject = simplify(self.subject)
        return type(self)(subject)


class Sin(Trig):
    def __repr__(self):
        return f"sin({self.subject})"

    def forward(self):
        return Trigonometric.sin(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return subject.cos()

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Cos(Trig):
    def __repr__(self):
        return f"cos({self.subject})"

    def forward(self):
        return Trigonometric.cos(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        return subject - self.subject.sin()

    def gradients(self, loss):
        return gradients(self.subject, loss * -self.subject.sin())


class Asin(Trig):
    def __repr__(self):
        return f"asin({self.subject})"

    def forward(self):
        return Trigonometric.asin(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return (1 - (subject**2))**-0.5

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Acos(Trig):
    def __repr__(self):
        return f"acos({self.subject})"

    def forward(self):
        return Trigonometric.acos(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return -((1 - subject**2)**-0.5)

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Sinh(Trig):
    def __repr__(self):
        return f"sinh({self.subject})"

    def forward(self):
        return Trigonometric.sinh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return subject.cosh()

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Cosh(Trig):
    def __repr__(self):
        return f"cosh({self.subject})"

    def forward(self):
        return Trigonometric.cosh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return subject.sinh()

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Asinh(Trig):
    def __repr__(self):
        return f"asinh({self.subject})"

    def forward(self):
        return Trigonometric.asinh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return (subject**2 + 1)**-0.5

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Acosh(Trig):
    def __repr__(self):
        return f"acosh({self.subject})"

    def forward(self):
        return Trigonometric.acosh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return ((subject**2) - 1)**-0.5

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Tan(Trig):
    def __repr__(self):
        return f"tan({self.subject})"

    def forward(self):
        return Trigonometric.tan(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return 1 / (subject.cos()**2)

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Tanh(Trig):
    def __repr__(self):
        return f"tanh({self.subject})"

    def forward(self):
        return Trigonometric.tanh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return 1 - subject.tanh()**2

    def gradients(self, loss):
        return gradients(self.subject, loss * self.backward())


class Atan(Trig):
    def __repr__(self):
        return f"atan({self.subject})"

    def forward(self):
        return Trigonometric.atan(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return 1 / (subject**2 + 1)

    def gradients(self, loss):
        return gradients(self.subject, loss * (self.subject**2 + 1)**(-1))


class Atanh(Trig):
    def __repr__(self):
        return f"atanh({self.subject})"

    def forward(self):
        return Trigonometric.atanh(self.subject)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable) if same_as(self.subject, variable) else self.subject
        return 1 / (1 - (subject**2))

    def gradients(self, loss):
        return gradients(self.subject, loss / (1 - self.subject**2))


class Dual(Operator):
    """A differentiable operator with two arguments"""

    def __init__(self, subject, args):
        if not is_numeric(subject):
            raise ValueError(f"{self.__class__.__name__} requires a Numeric subject, not {subject}")

        if not is_numeric(args):
            raise ValueError(f"{self.__class__.__name__} requires Numeric args, not {args}")

        Operator.__init__(self, subject, args)


class Log(Operator):
    def __repr__(self):
        return f"log({self.subject})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Numeric.log(self.subject)

    def backward(self, variable=None):
        if variable is None:
            return chain_rule(self, self.subject, self.args) / self.subject
        elif same_as(self.subject, variable):
            return 1 / derivative_of(self.subject, variable)
        else:
            return 1 / self.subject

    def gradients(self, loss):
        return gradients(self.subject, loss / self.subject)

    def simplify(self):
        subject = simplify(self.subject)
        args = simplify(self.args)
        return Log(subject, args)


class MatMul(Dual):
    def __repr__(self):
        return f"({self.subject}) @ ({self.args})"

    @property
    def shape(self):
        from ..shape import Shape
        return Shape(self.subject.shape[:-2]) + Shape((self.subject.shape[-1], self.args.shape[-2]))

    def forward(self):
        from ..collection.tensor import NDArray
        return NDArray.__matmul__(self.subject, self.args)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable, keepdims=True)
        arg = derivative_of(self.args, variable, keepdims=True)
        return (subject @ self.args) + (self.subject @ arg)

    def gradients(self, loss):
        # TODO: don't assume that self.subject.ndim == 2 and self.args.ndim == 2
        return (gradients(self.subject, loss @ self.args.transpose([1, 0])) +
                gradients(self.args, self.subject.transpose([1, 0]) @ loss))

    def simplify(self):
        subject = simplify(self.subject)
        args = simplify(self.args)

        from ..collection.tensor import NDArray

        if is_zero(subject) or is_zero(args):
            return zeros_like(self)
        elif isinstance(subject, NDArray) and isinstance(args, NDArray):
            return MatMul(subject, args)
        else:
            return self


class Pow(Dual):
    def __repr__(self):
        return f"({self.subject})**({self.args})"

    @property
    def shape(self):
        return self.subject.shape

    def forward(self):
        return Numeric.pow(self.subject, self.args)

    def backward(self, variable=None):
        if same_as(self.args, variable):
            return (self.subject**self.args) * self.subject.log()
        elif variable is None:
            return chain_rule(self, self.subject, self.args) * self.args * (self.subject**(self.args - 1))
        else:
            return self.args * (self.subject ** (self.args - 1))

    def gradients(self, loss):
        subject_grad = loss * self.args * self.subject**(self.args - 1)
        args_grad = loss * self.subject.log() * self.subject**self.args
        return gradients(self.subject, subject_grad) + gradients(self.args, args_grad)

    def simplify(self):
        subject = simplify(self.subject)
        args = simplify(self.args)

        if is_one(subject) or is_zero(args):
            return 1
        elif is_one(args):
            return subject

        return Pow(subject, args)


class DualBroadcast(Operator):
    @property
    def shape(self):
        if is_literal(self.subject):
            return self.args.shape
        elif is_literal(self.args):
            return self.subject.shape

        return self.subject.shape.broadcast(self.args.shape)


class LogicalAnd(DualBroadcast):
    def __repr__(self):
        return f"({self.subject}) AND ({self.args})"

    def forward(self):
        return Boolean.logical_and(self.subject, self.args)


class LogicalOr(DualBroadcast):
    def __repr__(self):
        return f"({self.subject}) OR ({self.args})"

    def forward(self):
        return Boolean.logical_or(self.subject, self.args)


class LogicalXor(DualBroadcast):
    def __repr__(self):
        return f"({self.subject}) XOR ({self.args})"

    def forward(self):
        return Boolean.logical_xor(self.subject, self.args)


class Add(DualBroadcast):
    def __repr__(self):
        return f"({self.subject}) + ({self.args})"

    def forward(self):
        return Numeric.add(self.subject, self.args)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return subject + arg

    def gradients(self, loss):
        return gradients(self.subject, loss) + gradients(self.args, loss)

    def simplify(self):
        subject = simplify(self.subject)
        args = simplify(self.args)

        if is_zero(subject) and is_zero(args):
            return 0
        if is_zero(subject):
            return args
        elif is_zero(args):
            return subject
        else:
            return Add(subject, args)


class Mul(DualBroadcast):
    def __repr__(self):
        return f"({self.subject}) * ({self.args})"

    def forward(self):
        return Numeric.mul(self.subject, self.args)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return (subject * self.args) + (self.subject * arg)

    def gradients(self, loss):
        return gradients(self.subject, self.args * loss) + gradients(self.args, self.subject * loss)

    def simplify(self):
        subject = simplify(self.subject)
        args = simplify(self.args)

        if is_zero(subject) or is_zero(args):
            return 0
        elif is_one(subject) and is_one(args):
            return 1
        elif is_one(subject):
            return args
        elif is_one(args):
            return subject
        else:
            return Mul(subject, args)


class Sub(DualBroadcast):
    def __repr__(self):
        return f"({self.subject}) - ({self.args})"

    def forward(self):
        return Numeric.sub(self.subject, self.args)

    def backward(self, variable=None):
        subject = derivative_of(self.subject, variable)
        arg = derivative_of(self.args, variable)
        return subject - arg

    def gradients(self, loss):
        return gradients(self.subject, loss) + gradients(self.args, -loss)

    def simplify(self):
        subject = simplify(self.subject)
        args = simplify(self.args)

        if is_zero(args):
            return subject
        elif same_as(subject, args):
            return 0
        else:
            return Sub(subject, args)


class Div(DualBroadcast):
    def __repr__(self):
        return f"({self.subject}) / ({self.args})"

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
        return gradients(self.subject, loss / self.args) + gradients(self.args, -self.subject * loss / self.args**2)

    def simplify(self):
        subject = simplify(self.subject)
        args = simplify(self.args)

        if is_zero(subject):
            return 0
        elif is_one(args):
            return subject
        else:
            return Div(subject, args)


def chain_rule(op, *args):
    """
    Compute the chain rule coefficient of the given :class:`Operator`.

    This function will return `1` if the given `numeric` is constant, or has only constant inputs.
    """

    if operator(op):
        op = operator(op)
    else:
        raise ValueError(f"cannot apply the chain rule to a constant {op}")

    args = [arg for arg in args if operator(arg)]

    if not args:
        return 1
    elif len(args) == 1:
        return derivative_of(args[0])
    else:
        return sum(derivative_of(op, arg) * derivative_of(arg) for arg in args)


def constant(numeric):
    """Return the given `numeric` state as a constant, i.e. not the result of a differentiable :class:`Operator`."""

    if is_literal(numeric):
        return numeric

    rtype = type(numeric)

    if not is_numeric(numeric):
        raise ValueError(f"a non-numeric state {numeric} (type {rtype}) cannot be a numeric constant")

    while operator(numeric):
        numeric = rtype(form=operator(numeric).forward())

    return numeric


def derivative_of(state_or_function, variable=None, keepdims=False):
    """
    Find the derivative of the given `state_or_function`.

    If a differentiable state is given, this will construct a new op to calculate it derivative, which can be
    a partial derivative if a `variable` is specified.

    If a differentiable function is given, a new callable function will be returned which computes its derivative.

    If `state_or_function` is not differentiable, a `TypeError` will be raised.
    """

    if callable(state_or_function):
        function = state_or_function
        if hasattr(function, "derivative"):
            return function.derivative(variable)
        else:
            raise ValueError(f"not a differentiable function: {function}")

    state = state_or_function

    if same_as(state, variable):
        # it's a partial derivative and this is the free variable
        return ones_like(state, keepdims)

    # TODO: it doesn't make sense to import from the ML package in the math package
    from ..ml.variable import Variable

    if isinstance(state, Variable):
        if variable is None:
            # it's not a partial derivative
            return ones_like(state, keepdims)
        else:
            # it's a partial derivative and this variable is held constant
            return zeros_like(state, keepdims)

    if is_constant(state):
        return zeros_like(state, keepdims)
    elif operator(state):
        d = operator(state).backward(variable)

        if keepdims:
            if same_as(d, 0):
                return zeros_like(state)
            elif same_as(d, 1):
                return ones_like(state)

        return d
    else:
        raise ValueError(f"the derivative of {state} is not defined")


def gradients(numeric, loss, variables=None):
    """
    Return the gradient of a `numeric` state with respect to the given `loss`.

    If one variable is given, one gradient will be returned, or a `KeyError` will be raised if not present in the graph.
    If a list of variables is given, a corresponding list of gradients will be returned.
    If no variables are given, a :class:`Gradients` object whose keys are the inputs of the graph will be returned.
    """

    if is_literal(numeric):
        grads = Gradients()
    elif operator(numeric):
        grads = operator(numeric).gradients(loss)
    elif is_constant(numeric):
        grads = Gradients({numeric: loss})
    elif is_numeric(numeric):
        raise ValueError(f"cannot compute gradients of {numeric} w/r/t {loss}")
    else:
        raise ValueError(f"not a numeric state: {numeric}")

    if variables is None:
        return grads

    if not isinstance(variables, (list, tuple)):
        if variables not in grads:
            raise KeyError(f"{variables} is not reachable from operator {numeric}")

        return grads[variables]

    missing = [var for var in variables if var not in grads]
    if missing:
        raise KeyError(f"not reachable by traversing the operator graph {numeric}: {missing}")

    return [grads[var] for var in variables]


def is_constant(numeric):
    """
    Return `False` if the given `numeric` state is the result of an :class:`Operator`, i.e. a differentiable function.
    """

    return operator(numeric) is None


def is_one(numeric):
    """Return `True` if the given `numeric` state is a constant with value one."""

    if same_as(numeric, 1):
        return True

    from ..collection.tensor import Dense, NDArray, Transform

    while isinstance(operator(numeric), Transform):
        numeric = operator(numeric).subject

    if same_as(numeric, 1):
        return True
    elif isinstance(numeric, NDArray) and same_as(numeric, Dense.ones_like(numeric)):
        return True

    return False


def is_zero(numeric):
    """Return `True` if the given `numeric` state is a constant with value zero."""

    if same_as(numeric, 0):
        return True

    from ..collection.tensor import Sparse, NDArray, Transform

    while isinstance(operator(numeric), Transform):
        numeric = operator(numeric).subject

    if same_as(numeric, 0):
        return True
    elif isinstance(numeric, NDArray) and same_as(numeric, Sparse.zeros_like(numeric)):
        return True

    return False


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

    if is_literal(state):
        return state

    if not is_numeric(state):
        raise TypeError(f"cannot simplify a non-numeric state: {state}")

    rtype = type(state)

    while operator(state):
        simplified = operator(state).simplify()
        if same_as(simplified, state):
            break

        state = simplified

    if is_literal(state):
        return state
    else:
        return rtype(form=state)


def shape_of(numeric):
    if isinstance(numeric, (bool, complex, float, int)):
        from ..shape import Shape
        return Shape(tuple())
    elif isinstance(numeric, Numeric):
        return numeric.shape
    else:
        raise ValueError(f"{numeric} has no shape")


def ones_like(state, keepdims=True):
    from ..collection.tensor import Dense
    from ..scalar.number import Number

    if isinstance(state, Number):
        return type(state)(form=1)
    elif is_literal(state) or not keepdims:
        return Number(1)
    else:
        cls = type(state) if hasattr(type(state), "ones_like") else Dense
        return cls.ones_like(state)


def zeros_like(state, keepdims=True):
    from ..collection.tensor import Sparse
    from ..scalar.number import Number

    if isinstance(state, Number):
        return type(state)(form=0)
    elif is_literal(state) or not keepdims:
        return Number(0)
    else:
        cls = type(state) if hasattr(type(state), "zeros_like") else Sparse
        return cls.zeros_like(state)


def debug_shape(numeric):
    if not hasattr(numeric, "shape"):
        raise ValueError(f"{numeric} has no shape")

    if is_literal(numeric.shape):
        print(f"the shape of {numeric} is {numeric.shape}")
        return

    print(f"{numeric} does not have a literal shape")

    op = operator(numeric)

    if not op:
        return

    from ..collection.tensor import NDArray

    if isinstance(op.subject, NDArray):
        debug_shape(op.subject)

    if isinstance(op.args, NDArray):
        debug_shape(op.args)
