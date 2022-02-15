from abc import abstractmethod, ABC

from tinychain.collection.tensor import Tensor
from tinychain.state.generic import Map, Tuple

EPS = 10**-6


class Activation(ABC):
    """A differentiable activation function for a :class:`Layer`."""

    @abstractmethod
    def forward(self, Z):
        """Compute the activation of the given :class:`Tensor`"""

    @abstractmethod
    def backward(self, Z):
        """Compute the partial differential of this function"""

    def std_initializer(self, input_size, output_size):
        """Calculate the optimal initial standard deviation for the inputs to this :class:`Activation`"""

        return (input_size * output_size)**0.5


class Identity(Activation):
    """An activation function which simply forwards its inputs"""

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def std_initializer(self, input_size, output_size):
        return 1.0 * (input_size*output_size)**(-0.5)


class Sigmoid(Activation):
    """Sigmoid activation function"""

    def forward(self, x):
        return 1 / (1 + (-x).exp())

    def backward(self, x):
        sig = self.forward(x)
        return sig * (1 - sig)

    def std_initializer(self, input_size, output_size):
        return 1.0 * (2 / (input_size + output_size))**0.5


#TODO: remove when automatic differentiation is implemented
class Tanh(Activation):
    """Hyperbolic tangent activation function"""

    def forward(self, x):
        return x.tanh()

    def backward(self, x):
        return 1 - self.forward(x)**2

    def std_initializer(self, input_size, output_size):
        return (5 / 3) * (2 / (input_size + output_size))**0.5


class ReLU(Activation):
    """ReLU activation function"""

    def forward(self, Z):
        return Z * (Z > 0)

    def backward(self, Z):
        return Z > 0

    def std_initializer(self, input_size, output_size):
        return (2**(0.5)) * (2 / (input_size + output_size))**0.5


class Differentiable(object):
    """A :class:`Differentiable` machine learning model, which can be used in the composition of a more complex model"""

    @classmethod
    @property
    @abstractmethod
    def shape(cls):
        """
        Return the shape of this gradient.

        This can be a :class:`Tuple` of :class:`U64` dimensions (for a :class:`Tensor`) or a Python `list` or `dict`
        (for a more complex trainable data structure like a :class:`NeuralNet`).
        """

    @abstractmethod
    def forward(self, inputs):
        """Evaluate this :class:`Differentiable` with respect to the given `inputs`."""

    @abstractmethod
    def backward(self, inputs, loss):
        """
        Compute the gradient of this :class`Differentiable` with respect to the given `inputs` and `loss`.

        Returns a tuple `(loss, diffed_params)` where `loss` is the loss to propagate further backwards and
        `diffed_params` is a flattened list of the :class:`DiffedParameter` s for an :class:`Optimizer` to optimize.
        """

    @abstractmethod
    def get_param_list(self):
        """Return the parameters of this :class:`Differentiable` as a `List[Parameter]` of :class:`Layer` s"""

        return []

    @abstractmethod
    def write(self, new_values):
        """
        Overwrite the values of this :class:`Differentiable` with the given `new_values`.

        `new_values` must have the same shape as this :class:`Differentiable`.
        """

        shape = self.shape
        if isinstance(shape, dict):
            return {name: self[name].write(new_values[name]) for name in shape}
        elif isinstance(shape, list) or isinstance(shape, tuple):
            return [self[i].write(new_values[i]) for i in range(len(shape))]
        else:
            raise NotImplementedError(f"{self.__class__} needs a `write` method")


class Layer(Map, Differentiable):
    """A :class:`Layer` in a :class:`NeuralNet`"""

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass


class NeuralNet(Tuple, Differentiable):
    """A neural network comprising a :class:`Tuple` of :class:`Layers`"""

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass


class Parameter:
    """A trainable :class:`Parameter` in a differentiable ML model."""

    def __init__(self, name: str, value: Tensor):
        self.name = name
        self.value = value

    @classmethod
    def create(cls, name: str, value: Tensor):
        """Create a new :class:`Parameter` with the given `name` and `value`"""

        return cls(name=name, value=value)


class DiffedParameter(Parameter):
    """Helper class to provide structured information about the differentiation of a `Parameter`."""

    def __init__(self, name: str, value: Tensor, grad: Tensor):
        super().__init__(name, value)
        self.grad = grad

    @classmethod
    def create(cls, name: str, value: Tensor, grad: Tensor):
        """Create a new :class:`DiffedParameter` with the given `name` and `value`"""

        return cls(name=name, value=value, grad=grad)
