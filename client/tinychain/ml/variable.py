from ..collection import tensor


# TODO: support Sparse and Number variable types
class Variable(tensor.Dense):
    """A trainable variable in a machine learning model."""

    def update(self, delta):
        """Decrement the value of this `Variable` by `delta`."""

        return self.write(self - delta)
