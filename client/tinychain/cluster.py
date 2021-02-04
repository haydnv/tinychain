import abc
import json

from . import reflect

from .state import State, Meta
from .util import *


class MetaCluster(Meta):
    def __json__(cls):
        instance = cls()
        instance.configure()
        return to_json(form_of(reflect.Instance(instance)))

    def __str__(cls):
        return json.dumps(to_json(cls), indent=4)


class Cluster(metaclass=MetaCluster):
    __uri__ = URI("/cluster")

    @abc.abstractmethod
    def configure(self):
        pass

