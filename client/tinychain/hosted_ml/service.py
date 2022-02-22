from tinychain.app import Library

from .activation import Activation, Sigmoid
from .nn import Layer, ConvLayer, DNNLayer, NeuralNet, Sequential
from . import LIB_URI


class ML(Library):
    __uri__ = LIB_URI

    @staticmethod
    def exports():
        return [
            Activation,
            Sigmoid,
            Layer,
            NeuralNet,
        ]

    @staticmethod
    def provides():
        return [ConvLayer, DNNLayer, Sequential]
