from ..service import library_uri, Library

from .constants import NS, VERSION
from .nn import NAME as NN, ConvLayer, DNN, Layer, Linear, NeuralNet, Sequential
from .optimizer import NAME as OPTIMIZER, Adam, GradientDescent, Optimizer


class NeuralNets(Library):
    __uri__ = library_uri(None, NS, NN, VERSION)

    # models (headers)
    Layer = Layer
    NeuralNet = NeuralNet

    # dynamic models (implementations)
    ConvLayer = ConvLayer
    DNN = DNN
    Linear = Linear
    Sequential = Sequential


class Optimizers(Library):
    __uri__ = library_uri(None, NS, OPTIMIZER, VERSION)

    # models (headers)
    Optimizer = Optimizer

    # dynamic models (implementations)
    Adam = Adam
    GradientDescent = GradientDescent
