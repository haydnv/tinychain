import abc

from .reflect import gen_headers
from .state import State
from .util import ref as get_ref, URI


class Cluster(object):
    __uri__ = URI("/cluster")

    @classmethod
    def __use__(cls):
        instance = cls()
        gen_headers(instance)
        return instance

    def __init__(self, ref=None):
        if ref is None:
            self.__ref__ = get_ref(self.__class__)
        else:
            self.__ref__ = ref

        self.configure()

    @abc.abstractmethod
    def configure(self):
        pass


