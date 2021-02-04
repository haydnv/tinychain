import abc

from .state import State
from .util import URI


class Cluster(object):
    __uri__ = URI("/cluster")

    def __init__(self, ref=None):
        if ref is None:
            self.__ref__ = ref(self.__class__)
        else:
            self.__ref__ = ref

        self.configure()

    @abc.abstractmethod
    def configure(self):
        pass

