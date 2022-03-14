from ..math import OperatorRef


class Activation(OperatorRef):
    """A differentiable activation function for a :class:`Layer`."""

    def __init__(self, subject):
        OperatorRef.__init__(self, subject, None)

    @staticmethod
    def optimal_std(input_size, output_size):
        """Calculate the optimal initial standard deviation for the inputs to this :class:`Activation`"""

        return (input_size * output_size)**0.5


class Sigmoid(Activation):
    """Sigmoid activation function"""

    @staticmethod
    def optimal_std(input_size, output_size):
        return 1.0 * (2 / (input_size + output_size))**0.5

    def forward(self):
        # return 1 / (1 + (-self.subject).exp())
        return self.subject

    # def backward(self):
    #     sig = self.forward()
    #     return sig * (1 - sig)
