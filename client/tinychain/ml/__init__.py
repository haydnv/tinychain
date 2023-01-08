"""Machine learning utilities such as a standard definition of a :class:`NeuralNet` and :class:`Optimizer`"""

from .activation import sigmoid, softmax, relu
from .constants import NS, VERSION
from .service import NeuralNets, Optimizers
from .variable import Variable
