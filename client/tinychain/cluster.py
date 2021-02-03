import json

from .util import to_json


class Cluster(object):
    PATH = None

    def __json__(self):
        return None

    def __str__(self):
        return json.dumps(to_json(self), indent=4)

