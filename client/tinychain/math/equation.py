import logging

from ..collection.tensor import Tensor
from ..scalar.ref import deref, hex_id, same_as, Op

from .operator import operator


class Function(object):
    def __init__(self, result):
        self.result = result

        self.nodes = {}
        self.edges = []

        self._traverse()

    def _traverse(self):
        assert not self.nodes
        assert not self.edges

        unvisited = self._unvisited()
        if not unvisited:
            logging.info(f"cannot traverse {self.result} since it is disconnected from any operator graph")

        while unvisited:
            node = unvisited.pop(0)
            op = operator(node)

            if operator(op.subject):
                self.edges.append((node, op.subject))
                if hex_id(op.subject) not in self.nodes:
                    unvisited.append(op.subject)

            if operator(op.args):
                self.edges.append((node, op.args))
                if hex_id(op.args) not in self.nodes:
                    unvisited.append(op.args)

            self.nodes[hex_id(node)] = node

    def _unvisited(self):
        if isinstance(self.result, (list, tuple)):
            unvisited = list(self.result)
        elif isinstance(self.result, dict):
            unvisited = list(self.result.values())
        elif isinstance(deref(self.result), Op):
            unvisited = [self.result]

        return [state for state in unvisited if operator(state)]

    def dedupe(self):
        canon = {}
        for node_id, node in self.nodes.items():
            assert same_as(node, node)
            dupes = [v for v in self.nodes.values() if same_as(node, v)]
            canon[node_id] = self.nodes[min(hex_id(n) for n in dupes)]

        visited = {}
        unvisited = self._unvisited()
        while unvisited:
            node = unvisited.pop(0)
            op = operator(node)

            if operator(op.subject):
                if not hex_id(op.subject) in visited:
                    unvisited.append(op.subject)

                op.subject = canon[hex_id(op.subject)]

            if operator(op.args):
                if not hex_id(op.args) in visited:
                    unvisited.append(op.args)

                op.args = canon[hex_id(op.args)]

            visited[hex_id(node)] = node

        return self.result

    def memoize(self):
        counts = {node_id: 0 for node_id in self.nodes}
        for f, _t in self.edges:
            counts[hex_id(f)] += 1

        copies = {
            node_id: self.nodes[node_id].copy() for node_id, count in counts.items()
            if count > 1 and isinstance(self.nodes[node_id], Tensor)
        }

        visited = {}
        unvisited = self._unvisited()
        while unvisited:
            node = unvisited.pop(0)
            op = operator(node)

            if hex_id(op.subject) in copies:
                if hex_id(op.subject) not in visited:
                    unvisited.append(op.subject)

                op.subject = copies[hex_id(op.subject)]

            if hex_id(op.args) in copies:
                if hex_id(op.args) not in visited:
                    unvisited.append(op.args)

                op.args = copies[hex_id(op.args)]

            visited[hex_id(node)] = node

        return self.result

    def optimize(self):
        """
        Perform an in-place optimization of this :class:`Function`.

        This will consolidate logically equivalent operations and memoize intermediate states where needed.
        """

        self.dedupe()
        self.memoize()

        return self.result
