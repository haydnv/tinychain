from ..collection import tensor
from ..util import form_of, hex_id


# TODO: support Sparse and Number variable types
class Variable(tensor.Dense):
    """A trainable variable in a machine learning model."""

    def __getitem__(self, bounds):
        return Variable(tensor.Dense.__getitem__(self, bounds))

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

    def update(self, delta):
        """Decrement the value of this `Variable` by `delta`."""

        if isinstance(form_of(self), tensor.Transform):
            return form_of(self).update(delta)
        else:
            return self.write(self - delta)

    def reshape(self, shape):
        shape = tuple(dim for dim in form_of(shape))
        return Variable.expect(shape, self.dtype)(Reshape(self, shape))

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
    def gradients(self, loss):
        return {hex_id(self.subject): loss.reshape(self.subject.shape)}

    def update(self, delta):
        return self.subject.update(delta.reshape(self.subject.shape))


class Flip(Transform, tensor.Flip):
    def gradients(self, loss):
        return {hex_id(self.subject): loss.flip(self.args)}

    def update(self, delta):
        return self.subject.update(delta.flip(self.args))


class Transpose(Transform, tensor.Transpose):
    def gradients(self, loss):
        return {hex_id(self.subject): loss.transpose(self.inverse_permutation)}

    def update(self, delta):
        return self.subject.update(delta.transpose(self.inverse_permutation))


class Reshape(Transform, tensor.Reshape):
    def gradients(self, loss):
        old_shape = self.subject.shape
        return {hex_id(self.subject): loss.reshape(old_shape)}

    def update(self, delta):
        return self.subject.update(delta.reshape(self.subject.shape))
