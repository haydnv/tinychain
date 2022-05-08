from ..util import hex_id, same_as

from .operator import operator


def optimize(state):
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
