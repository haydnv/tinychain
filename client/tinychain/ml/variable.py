from ..collection import tensor
from ..util import form_of, hex_id
from ..scalar.number import Number
from ..scalar.ref import After, MethodSubject


# TODO: support Sparse and Number variable types

class Variable(tensor.Dense):
    """A trainable variable in a machine learning model."""

    def cast(self, number_type):
        return Variable(tensor.Dense.cast(self, number_type))

    def expand_dims(self, axis=None):
        axis = form_of(axis)
        assert axis == int(axis)

        if axis is None:
            axis = self.ndim
        elif axis < 0:
            axis = self.ndim + axis

        shape = list(form_of(self.shape))
        shape.insert(axis, 1)

        raise Variable.expect(shape, self.dtype)(Expand(self, axis))

    def flip(self, axis):
        return self.__class__(form=Flip(self, axis))

    def invert(self, loss):
        if isinstance(form_of(self), Transform):
            return form_of(self).invert(loss)
        else:
            return {hex_id(self): loss}

    def update(self, delta):
        """Decrement the value of this `Variable` by `delta`."""

        return self.write(self - delta)

    def reshape(self, shape):
        shape = tuple(dim for dim in form_of(shape))
        return Variable.expect(shape, self.dtype)(Reshape(self, shape))

    def slice(self, bounds):
        parent = self
        bounds = tensor.handle_bounds(bounds)

        class WritableView(Variable):
            def write(self, value):
                return parent._put("", bounds, value)

        return WritableView(Slice(self, bounds))

    def transpose(self, permutation=None):
        if permutation is None:
            permutation = tuple(reversed(tuple(range(form_of(self.ndim)))))
        else:
            permutation = tuple(x for x in form_of(permutation))

        shape = tuple(self.shape[x] for x in permutation)
        return Variable.expect(shape, self.dtype)(Transpose(self, permutation))


class Transform(tensor.Transform):
    def __init__(self, subject, args):
        if not isinstance(subject, Variable):
            raise TypeError(f"the subject of a Variable transform must be a Variable, not {subject}")

        tensor.Transform.__init__(self, subject, args)

    def update(self):
        raise NotImplementedError(f"{self.__class__}.update")


class Expand(Transform, tensor.Expand):
    def invert(self, loss):
        if isinstance(loss, Number):
            return self.subject.invert(loss)

        return self.subject.invert(loss.reshape(self.subject.shape))


class Flip(Transform, tensor.Flip):
    def invert(self, loss):
        if isinstance(loss, Number):
            return self.subject.invert(loss)

        return self.subject.invert(loss.flip(self.args))


class Transpose(Transform, tensor.Transpose):
    def invert(self, loss):
        if isinstance(loss, Number):
            return self.subject.invert(loss)

        return self.subject.invert(loss.transpose(self.inverse_permutation))


class Reshape(Transform, tensor.Reshape):
    def invert(self, loss):
        if isinstance(loss, Number):
            return self.subject.invert(loss)

        return self.subject.invert(loss.reshape(self.subject.shape))


class Slice(Transform):
    def forward(self):
        return tensor.NDArray.slice(self.subject, self.args)

    def gradients(self, loss):
        if isinstance(loss, Number):
            return Transform.gradients(loss)

        grad = tensor.Dense.zeros_like(self.subject)  # TODO: this should support Sparse tensors as well
        grad = tensor.Dense(After(grad[self.args].write(loss), MethodSubject(grad, "")))
        return self.subject.invert(grad)
