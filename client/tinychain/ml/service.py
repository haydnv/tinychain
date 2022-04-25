from ..app import Library
from ..decorators import post

from .constants import LIB_URI
from .nn import ConvLayer, DNN, Layer, Linear, NeuralNet, Sequential
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
    DNN = DNN
    GradientDescent = GradientDescent
    Linear = Linear
    Sequential = Sequential

    @post
    def train(self, optimizer: Optimizer, inputs):
        return optimizer.train(1, inputs)
