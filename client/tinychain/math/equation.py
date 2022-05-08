from ..collection.tensor import Tensor
from ..util import hex_id, same_as

from .operator import operator


def dedupe(state):
    """De-duplicate results in the given operator graph in-place."""

    if not operator(state):
        raise ValueError(f"a math function requires an Operator, not {state}")

    visited = {}
    unvisited = [state]
    while unvisited:
        node = unvisited.pop(0)
        op = operator(node)

        if operator(op.subject) and hex_id(op.subject) not in visited:
            unvisited.append(op.subject)

        if operator(op.args) and hex_id(op.args) not in visited:
            unvisited.append(op.args)

        visited[hex_id(node)] = node

    canon = {}
    for node_id, node in visited.items():
        dupes = [v for v in visited.values() if same_as(node, v)]
        canon[node_id] = visited[min(hex_id(n) for n in dupes)]

    visited = {}
    unvisited = [state]
    while unvisited:
        node = operator(unvisited.pop(0))

        if operator(node.subject):
            visited[hex_id(node.subject)] = node
            operator(node).subject = canon[hex_id(node.subject)]

        if operator(node.args):
            visited[hex_id(node.subject)] = node
            operator(node).args = canon[hex_id(node.args)]

    return state


def memoize(state):
    """Memoize intermediate states with more than one edge in the given operator graph, in-place."""

    if not operator(state):
        raise ValueError(f"a math function requires an Operator, not {state}")

    edges = []

    visited = {}
    unvisited = [state]
    while unvisited:
        node = unvisited.pop(0)
        op = operator(node)

        if operator(op.subject):
            edges.append((node, op.subject))
            if hex_id(op.subject) not in visited:
                unvisited.append(op.subject)

        if operator(op.args):
            edges.append((node, op.args))
            if hex_id(op.args) not in visited:
                unvisited.append(op.args)

        visited[hex_id(node)] = node

    counts = {node_id: 0 for node_id in visited}
    for f, _t in edges:
        counts[hex_id(f)] += 1

    copies = {
        node_id: visited[node_id].copy() for node_id, count in counts.items()
        if count > 1 and isinstance(visited[node_id], Tensor)
    }

    visited = {}
    unvisited = [state]
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

    return state


def optimize(state):
    """
    Perform an in-place optimization of the given differentiable `state`.

    This will consolidate logically equivalent operations and memoize intermediate states where needed.
    """

    dedupe(state)
    memoize(state)
    return state
