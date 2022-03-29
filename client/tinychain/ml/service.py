from ..app import Library

from .nn import Layer, NeuralNet
from .optimizer import Optimizer
from . import LIB_URI


class ML(Library):
    __uri__ = LIB_URI

    @staticmethod
    def exports():
        return [
            Layer,
            NeuralNet,
            Optimizer,
        ]
