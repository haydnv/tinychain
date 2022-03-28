from __future__ import annotations

import numpy as np


class GraphNode:
    name: str

    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    @property
    def value(self) -> np.ndarray:
        return self._value

    def backward(self) -> np.ndarray:
        raise NotImplementedError()


class TensorPlaceholderGraphNode(GraphNode):
    name: str = 'tensor_placeholder'

    def backward(self) -> np.ndarray:
        return self._value


class TensorAddGraphNode(GraphNode):
    name: str = 'tensor_add'

    def __init__(self, value: np.ndarray, left: GraphNode, right: GraphNode) -> None:
        super().__init__(value)
        self._left = left
        self._right = right

    def backward(self) -> np.ndarray:
        return self._left.backward() + self._right.backward()


class TensorMulGraphNode(GraphNode):
    name: str = 'tensor_mul'

    def __init__(self, value: np.ndarray, left: GraphNode, right: GraphNode) -> None:
        super().__init__(value)
        self._left = left
        self._right = right

    def backward(self) -> np.ndarray:
        return self._left.backward() * self._right.value + self._left.value * self._right.backward()


class Operator:

    def __init__(self, graph: GraphNode):
        self._graph = graph

    @classmethod
    def from_tensor(cls, value: np.ndarray) -> Operator:
        return Operator(TensorPlaceholderGraphNode(value))

    @property
    def graph(self) -> GraphNode:
        return self._graph

    def __add__(self, other: Operator) -> Operator:
        return Operator(
            graph=TensorAddGraphNode(self._graph.value + other.graph.value, self._graph, other.graph))

    def __mul__(self, other: Operator) -> Operator:
        return Operator(
            graph=TensorMulGraphNode(self._graph.value * other.graph.value, self._graph, other.graph))

    def backward(self) -> np.ndarray:
        return self._graph.backward()


a = Operator.from_tensor(np.array([1, 0, 0, 1]).reshape((2, 2)))
b = Operator.from_tensor(np.array([1, 0, 0, 1]).reshape((2, 2)))
c = Operator.from_tensor(np.array([2, 1, 4, 3]).reshape((2, 2)))

d = (a + b) * c
print(d.backward())
