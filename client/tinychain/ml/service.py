from ..app import Library

from .constants import LIB_URI
from .nn import ConvLayer, DNN, DNNLayer, Layer, Linear, NeuralNet, Sequential
from .optimizer import Adam, GradientDescent, Optimizer


class ML(Library):
    __uri__ = LIB_URI

    # models (headers)
    Layer = Layer
    NeuralNet = NeuralNet
    Optimizer = Optimizer

    # dynamic models (implementations)
    Adam = Adam
    ConvLayer = ConvLayer
    DNNLayer = DNNLayer
    DNN = DNN
    GradientDescent = GradientDescent
    Linear = Linear
    Sequential = Sequential
