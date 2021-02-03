import json

from .util import *


class Cluster(object):
    PATH = None

    def __init__(self, spec):
        self.__spec__ = spec

    def __str__(self):
        return json.dumps(to_json(self), indent=4)

