from ..app import Library

from .constants import LIB_URI
from .nn import ConvLayer, DNN, Layer, Linear, NeuralNet, Sequential
from .optimizer import Adam, GradientDescent, Optimizer


class ML(Library):
    URI = LIB_URI

    __uri__ = URI

    # models (headers)
    Layer = Layer
    NeuralNet = NeuralNet
    Optimizer = Optimizer

    # dynamic models (implementations)
    Adam = Adam
    ConvLayer = ConvLayer
    DNN = DNN
    GradientDescent = GradientDescent
    Linear = Linear
    Sequential = Sequential
